#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import argparse
import concurrent.futures
import json
import logging
import os
import shutil
import subprocess
from pathlib import Path

import deep_sdf
import deep_sdf.workspace as ws


def filter_classes_glob(patterns, classes):
    import fnmatch

    passed_classes = set()
    for pattern in patterns:

        passed_classes = passed_classes.union(
            set(filter(lambda x: fnmatch.fnmatch(x, pattern), classes))
        )

    return list(passed_classes)


def filter_classes_regex(patterns, classes):
    import re

    passed_classes = set()
    for pattern in patterns:
        regex = re.compile(pattern)
        passed_classes = passed_classes.union(set(filter(regex.match, classes)))

    return list(passed_classes)


def filter_classes(patterns, classes):
    if patterns[0] == "glob":
        return filter_classes_glob(patterns, classes[1:])
    elif patterns[0] == "regex":
        return filter_classes_regex(patterns, classes[1:])
    else:
        return filter_classes_glob(patterns, classes)


def process_mesh(mesh1_filepath, mesh2_filepath, target1_filepath, target2_filepath, executable, additional_args):
    logging.info(mesh1_filepath + " --> " + target1_filepath)
    logging.info(mesh2_filepath + " --> " + target2_filepath)
    command = [
        executable, 
        "--hand", mesh1_filepath, 
        "--obj", mesh2_filepath,
        "--outhand", target1_filepath,
        "--outobj", target2_filepath
    ] + additional_args

    print()
    print(" ".join(elem for elem in command))
    print()

    subproc = subprocess.Popen(command, stdout=subprocess.DEVNULL)
    subproc.wait()


def append_data_source_map(data_dir, name, source):

    data_source_map_filename = ws.get_data_source_map_filename(data_dir)

    print("data sources stored to " + data_source_map_filename)

    data_source_map = {}

    if os.path.isfile(data_source_map_filename):
        with open(data_source_map_filename, "r") as f:
            data_source_map = json.load(f)

    if name in data_source_map:
        if not data_source_map[name] == os.path.abspath(source):
            raise RuntimeError(
                "Cannot add data with the same name and a different source."
            )

    else:
        data_source_map[name] = os.path.abspath(source)

        with open(data_source_map_filename, "w") as f:
            json.dump(data_source_map, f, indent=2)

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="Pre-processes data from a data source and append the results to "
        + "a dataset.",
    )
    arg_parser.add_argument(
        "--data_dir",
        "-d",
        dest="data_dir",
        required=True,
        help="The directory which holds all preprocessed data.",
    )
    arg_parser.add_argument(
        "--source",
        "-s",
        dest="source_dir",
        required=True,
        help="The directory which holds the data to preprocess and append.",
    )
    arg_parser.add_argument(
        "--threads",
        dest="num_threads",
        default=8,
        help="The number of threads to use to process the data.",
    )

    # export PANGOLIN_WINDOW_URI=headless://

    args = arg_parser.parse_args()

    executable = str(Path.cwd() / "scripts/sample_points/bin/PreprocessMesh")
    extension = ".npz"
    transform_filename = "transform.csv"
    commands = []

    target_dir = Path(args.data_dir)
    source_dir = Path(args.source_dir)
    for obj_dir in source_dir.iterdir():

        input_partA = obj_dir / "partA.obj"
        input_partB = obj_dir / "partB.obj"

        output_dir = target_dir / obj_dir.name
        output_partA = output_dir / ("partA" + extension)
        output_partB = output_dir / ("partB" + extension)

        output_dir.mkdir(parents=True, exist_ok=True)

        # copy over transformation
        shutil.copy(obj_dir / transform_filename, output_dir / transform_filename)

        commands.append(
            (
                str(Path.cwd() / input_partA),
                str(Path.cwd() / input_partB),
                str(Path.cwd() / output_partA),
                str(Path.cwd() / output_partB),
            )
        )

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=int(args.num_threads)
    ) as executor:

        for (
            input_partA,
            input_partB,
            output_partA,
            output_partB
        ) in commands:
            executor.submit(
                process_mesh,
                input_partA,
                input_partB,
                output_partA,
                output_partB,
                executable,
                [],
            )

        executor.shutdown()