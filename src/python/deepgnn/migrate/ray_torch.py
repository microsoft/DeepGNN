# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Script to migrate torch example to use Ray Train."""
import os
import shutil
import argparse
from pathlib import Path
import pasta
from pasta.augment import rename


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Migrate to new versions of DeepgNN.")
    parser.add_argument(
        "--main_path", type=str, required=True, help="Directory to migrate."
    )
    args = parser.parse_args()

    dirname = os.path.dirname(args.main_path)
    if dirname == "":
        dirname = "."
    shutil.copyfile(os.path.join(os.path.dirname(__file__), "data", "ray_util.py"), os.path.join(dirname, "ray_util.py"))

    with open(args.main_path, "r") as file:
        raw_input = file.read()

    tree = pasta.parse(raw_input)

    # TODO remove all deepgnn.pytorch.training
    rename.rename_external(
        tree, "deepgnn.pytorch.training.util.disable_infini_band", "deepgnn.pytorch.util.disable_infini_band"
    )
    rename.rename_external(
        tree, "deepgnn.pytorch.training.run_dist", "ray_util.run_ray"
    )

    raw_output = pasta.dump(tree)

    with open(args.main_path, "w") as file:
        file.write(raw_output)
