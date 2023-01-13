# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Script to migrate torch example to use Ray Train."""
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
        help="Path to main file that contains run_dist.",
    )
    parser.add_argument(
        "--hvd",
        action="store_true",
        default=False,
        help="Convert to Horovod trainer instead of standard DDP trainer.",
    )
    args = parser.parse_args()

    with open(args.main_path, "r") as file:
        raw_input = file.read()

    tree = pasta.parse(raw_input)

    # TODO remove all deepgnn.pytorch.training
    rename.rename_external(
        tree,
        "deepgnn.pytorch.training.util.disable_infini_band",
        "deepgnn.pytorch.util.disable_infini_band",
    )
    if args.hvd:
        rename.rename_external(
            tree, "deepgnn.pytorch.training.run_dist", "deepgnn.pytorch.common.horovod_train.run_ray"
        )
    else:
        rename.rename_external(
            tree, "deepgnn.pytorch.training.run_dist", "deepgnn.pytorch.common.ray_train.run_ray"
        )

    raw_output = pasta.dump(tree)

    raw_output = raw_output.replace(
        "from deepgnn.pytorch.common.utils import set_seed", ""
    )
    raw_output = raw_output.replace(", set_seed", "")
    raw_output = raw_output.replace("set_seed,", "")
    raw_output = raw_output.replace(
        """    if args.seed:
        set_seed(args.seed)""",
        "",
    )
    raw_output = raw_output.replace("set_seed(args.seed)", "")

    with open(args.main_path, "w") as file:
        file.write(raw_output)
