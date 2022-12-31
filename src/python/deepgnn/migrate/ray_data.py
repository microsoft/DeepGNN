# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Script to migrate dataset usage to use Ray Data."""
import os
import argparse
import pasta
from pasta.augment import rename


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Migrate to new versions of DeepgNN.")
    parser.add_argument(
        "--main_path",
        type=str,
        required=True,
        help="Path to main file that contains TorchDeepGNNDataset.",
    )
    args = parser.parse_args()

    with open(args.main_path, "r") as file:
        raw_input = file.read()

    tree = pasta.parse(raw_input)

    rename.rename_external(
        tree, "deepgnn.pytorch.common.dataset.TorchDeepGNNDataset", "ray_dataset"
    )

    raw_output = pasta.dump(tree)

    raw_output = raw_output.replace("set_seed(args.seed)", "")

    with open(args.main_path, "w") as file:
        file.write(raw_output)
