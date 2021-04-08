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
import copy

# Headless mesh to sdf
os.environ['PYOPENGL_PLATFORM'] = 'egl'
import trimesh

import numpy as np
import torch
import torch.utils.data as data_utils
import torch.multiprocessing as mp
from pytorch3d.transforms import quaternion_apply
from pytorch3d.transforms.rotation_conversions import quaternion_to_matrix
from torch import distributions as dist
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, models, transforms

import networks.model as arch
import utils

import open3d as o3d
import plotly.graph_objects as go

device = torch.device('cuda')

def get_spec_with_default(specs, key, default):
    try:
        return specs[key]
    except KeyError:
        return default

def shape_assembly_visualization_function(part1, part2, gt_transform, encoder_decoder, subsample, scene_per_batch):

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

        mesh1 = o3d.io.read_triangle_mesh(part1["mesh_filename"][0])
        mesh2 = o3d.io.read_triangle_mesh(part2["mesh_filename"][0])

        mesh1.compute_vertex_normals()
        mesh2.compute_vertex_normals()

        mesh1.paint_uniform_color([1, 0.706, 0])
        mesh2.paint_uniform_color([0.706, 1, 0])

        o3d.visualization.draw_geometries([mesh1, mesh2])

        new_mesh = copy.deepcopy(mesh1)
        gt_transform = gt_transform[0].numpy()
        new_mesh.transform(gt_transform)
        new_mesh.rotate(gt_transform[0:3,0:3].T)
        # for i in range(len(new_mesh.vertices)):
        #     new_mesh.vertices[i] = new_mesh.vertices[i] - np.dot(gt_transform[0:3,0:3].T, gt_transform[0:3,3])
        #     new_mesh.vertices[i] = np.dot(gt_transform[0:3,0:3].T, new_mesh.vertices[i])
        
        new_mesh.compute_vertex_normals()

        o3d.visualization.draw_geometries([new_mesh, mesh2])

        predicted_rotation = quaternion_to_matrix(predicted_rotation)

        predicted_rotation = predicted_rotation.detach()
        predicted_translation = predicted_translation.detach()

        transformation = np.zeros((4,4))
        transformation[3,3] = 1

        for point_ind in range(subsample):
            
            transformation[:3,:3] = predicted_rotation[point_ind].numpy()
            transformation[:3,3] = predicted_translation[point_ind].numpy()

            # Rotate mesh1 to be aligned with mesh2
            new_mesh = copy.deepcopy(mesh1)

            new_mesh = new_mesh.transform(transformation)
            o3d.visualization.draw_geometries([new_mesh, mesh2])

        return

def run_visualization(experiment_directory, checkpoint):

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

    sdf_dataset = utils.data.SDFAssemblySamples(data_source, train_split, subsample)
    sdf_val_dataset = utils.data.SDFAssemblySamples(data_source, val_split, subsample)

    sdf_loader = data_utils.DataLoader(
        sdf_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=num_data_loader_threads,
        drop_last=True
    )
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

    model_epoch = utils.misc.load_model_parameters(
        experiment_directory, checkpoint, encoder_decoder
    )

    logging.info("Loaded weights!")

    total_validation_error = 0
    # Run validation
    for i, (
        part1,
        part2,
        gt_transformed_part1_points,
        gt_transform) in enumerate(sdf_val_loader):

            shape_assembly_visualization_function(part1, part2, gt_transform, encoder_decoder, subsample, scene_per_batch)

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
    arg_parser.add_argument(
        "--checkpoint",
        "-c",
        dest="checkpoint",
        required=True,
        help="The checkpoint to load weights from.",
    )

    utils.add_common_args(arg_parser)

    args = arg_parser.parse_args()

    utils.configure_logging(args)
    
    run_visualization(args.experiment_directory, args.checkpoint)