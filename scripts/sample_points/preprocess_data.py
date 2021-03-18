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


def process_mesh(mesh_filepath, target_filepath, executable, additional_args):
    logging.info(mesh_filepath + " --> " + target_filepath)
    command = [executable, "-m", mesh_filepath, "-o", target_filepath] + additional_args

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

    args = arg_parser.parse_args()

    executable = str(Path.cwd() / "scripts/sample_points/bin/PreprocessMesh")
    extension = ".npz"
    transform_filename = "transform.csv"
    commands = []

    target_dir = Path(args.data_dir)
    source_dir = Path(args.source_dir)
    for path in source_dir.rglob('*.obj'):
        relative = Path(str(path)[len(str(source_dir)):])
        target = Path(str(target_dir) + str(relative)[:-4] + extension)
        target.parent.mkdir(parents=True, exist_ok=True)

        # copy over transformation
        shutil.copy(path.parent / transform_filename, target.parent / transform_filename)

        commands.append(
            (
                str(path),
                str(target),
            )
        )

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=int(args.num_threads)
    ) as executor:

        for (
            mesh_source_path,
            target_filepath,
        ) in commands:
            executor.submit(
                process_mesh,
                mesh_source_path,
                target_filepath,
                executable,
                [],
            )

        executor.shutdown()