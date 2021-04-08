import warnings

warnings.filterwarnings('ignore',category=FutureWarning)
import json
import logging
import os
import signal
import sys
import time
from threading import Thread
from pathlib import Path

# Headless mesh to sdf
os.environ['PYOPENGL_PLATFORM'] = 'egl'
from mesh_to_sdf import mesh_to_sdf
import trimesh

import numpy as np
import torch
import torch.utils.data as data_utils
import torch.multiprocessing as mp
from pytorch3d.transforms import quaternion_apply
from torch import distributions as dist
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, models, transforms

import networks.model as arch
import utils

device = torch.device('cuda')

def get_spec_with_default(specs, key, default):
    try:
        return specs[key]
    except KeyError:
        return default

def shape_assembly_val_function(part1, part2, gt_transformation, encoder_decoder, subsample, scene_per_batch):

        samples = part2["sdf_samples"]

        samples.requires_grad = False

        sdf_data = (samples.to(device)).reshape(
            subsample * scene_per_batch, 5
        )
        
        xyzs = sdf_data[:, 0:3]
        sdf_gt_part1 = sdf_data[:, 3].unsqueeze(1)
        sdf_gt_part2 = sdf_data[:, 4].unsqueeze(1)
        
        part1_transform_vec = torch.cat((part1["center"], part1["quaternion"]), 1).to(device)

        _, _, predicted_translation, predicted_rotation = encoder_decoder(
                                                part1["surface_points"].to(device), 
                                                part2["surface_points"].to(device), 
                                                xyzs,
                                                part1_transform_vec
                                            )

        #apply the predicted transformation to the points
        #Should return Nx3 transformed points
        predicted_translation = predicted_translation.reshape((predicted_translation.shape[0], 1, 3))
        predicted_rotation = predicted_rotation.reshape((predicted_rotation.shape[0], 1, 4))
        predicted_rotation = predicted_rotation.to(device)
        part1["surface_points"] = part1["surface_points"].to(device)
        predicted_rotated_part1_points = quaternion_apply(predicted_rotation, part1["surface_points"])
        predicted_transformed_part1_interface_points = torch.add(predicted_rotated_part1_points,predicted_translation)

        translation_error_avg = predicted_translation - gt_transformation[:3,3]
        rotation_error = predicted_rotation - gt_transformation[:3,:3]
        # 
        mesh1 = trimesh.load(part1["mesh_filename"][0])
        mesh2 = trimesh.load(part2["mesh_filename"][0])

        part1["interface_points"] = part1["interface_points"].numpy().reshape((-1,3))[:50]
        part2["interface_points"] = part2["interface_points"].numpy().reshape((-1,3))[:50]

        sdf1 = mesh_to_sdf(mesh2, part1["interface_points"], surface_point_method="sample")
        sdf2 = mesh_to_sdf(mesh1, part2["interface_points"], surface_point_method="sample")

        # Use absolute values
        sdf1 = np.abs(sdf1)
        sdf2 = np.abs(sdf2)

        cardinality = len(sdf1)+len(sdf2)
        val_metric = (1/cardinality)*(sdf1.sum()+sdf2.sum())

        return val_metric

def run_validation(experiment_directory):

    specs = utils.misc.load_experiment_specifications(experiment_directory)

    data_source = Path(specs["DataSource"])
    train_split_file = specs["TrainSplit"]
    val_split_file = specs["ValSplit"]
    subsample = specs["SamplesPerScene"]
    scene_per_batch = specs["ScenesPerBatch"]
    latent_size = specs["LatentSize"]
    num_epochs = specs["NumEpochs"]
    num_data_loader_threads = get_spec_with_default(specs, "DataLoaderThreads", 8)
    val_split_file = get_spec_with_default(specs, "ValSplit", None)
    nb_classes = get_spec_with_default(specs["NetworkSpecs"], "num_class", 6)
    log_frequency = get_spec_with_default(specs, "LogFrequency", 5)
    log_frequency_step = get_spec_with_default(specs, "LogFrequencyStep", 100)

    with open(train_split_file, "r") as f:
        train_split = json.load(f)

    with open(val_split_file, "r") as f:
        val_split = json.load(f)

    sdf_val_dataset = utils.data.SDFAssemblySamples(data_source, val_split, subsample)

    sdf_val_loader = data_utils.DataLoader(
        sdf_val_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=num_data_loader_threads,
        drop_last=True
    )

    half_latent_size = int(latent_size/2)

    encoder_part1 = arch.ResnetPointnet(c_dim=half_latent_size, hidden_dim=256).to(device)
    encoder_part2 = arch.ResnetPointnet(c_dim=half_latent_size, hidden_dim=256).to(device)
    print("Point cloud encoder, each branch has latent size", half_latent_size)

    decoder = arch.ShapeAssemblyDecoder(
        latent_size, 
        **specs["NetworkSpecs"]
    )

    encoder_decoder = arch.ModelShapeAssemblyEncoderDecoderVAE(
        encoder_part1,
        encoder_part2,
        decoder,
        nb_classes,
        subsample
    )
    encoder_decoder = encoder_decoder.to(device)
    encoder_decoder.share_memory()

    writer = SummaryWriter(os.path.join(experiment_directory, 'log'))

    model_parameters_path = Path(experiment_directory) / "ModelParameters"
    for model_snapshot in model_parameters_path.iterdir():

        logging.info("Running validation with weights: " + model_snapshot.stem)

        model_epoch = utils.misc.load_model_parameters(
            experiment_directory, model_snapshot.stem, encoder_decoder
        )

        logging.info("Loaded weights!")

        total_validation_error = 0
        # Run validation
        for i, (
            part1,
            part2,
            gt_transformed_part1_points,
            gt_transform) in enumerate(sdf_val_loader):

            total_validation_error += shape_assembly_val_function(part1, part2, gt_transform, encoder_decoder, subsample, scene_per_batch)

        writer.add_scalar('validation_loss', total_validation_error / len(sdf_val_loader), model_epoch)

if __name__ == "__main__":

    import argparse

    arg_parser = argparse.ArgumentParser(description="Train a DeepSDF autodecoder")
    arg_parser.add_argument(
        "--experiment",
        "-e",
        dest="experiment_directory",
        required=True,
        help="The experiment directory. This directory should include "
        + "experiment specifications in 'specs.json', and logging will be "
        + "done in this directory as well.",
    )

    utils.add_common_args(arg_parser)

    args = arg_parser.parse_args()

    utils.configure_logging(args)
    
    run_validation(args.experiment_directory)