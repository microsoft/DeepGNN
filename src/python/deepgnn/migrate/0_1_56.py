# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Script to migrate to new versions of DeepGNN."""
import argparse
from pathlib import Path
import pasta
from pasta.augment import rename


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Migrate to new versions of DeepgNN.")
    parser.add_argument(
        "--script_dir", type=str, required=True, help="Directory to migrate."
    )
    args = parser.parse_args()

    for filename in Path(args.script_dir).glob("**/*.py"):
        with open(filename, "r") as file:
            raw_input = file.read()

        raw_input = raw_input.replace("FeatureType.FLOAT", "np.float32")
        raw_input = raw_input.replace("FeatureType.INT64", "np.int64")
        raw_input = raw_input.replace("FeatureType.BINARY", "np.uint8")

        tree = pasta.parse(raw_input)

        rename.rename_external(
            tree, "deepgnn.graph_engine.FeatureType", "numpy.dtype_temp"
        )

        raw_output = pasta.dump(tree)

        raw_output = raw_output.replace(
            "from numpy import dtype_temp", "import numpy as np"
        )
        raw_output = raw_output.replace("dtype_temp", "np.dtype")

        with open(filename, "w") as file:
            file.write(raw_output)
