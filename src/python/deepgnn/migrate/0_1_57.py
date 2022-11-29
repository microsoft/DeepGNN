# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Script to migrate to new versions of DeepGNN."""
import argparse
from pathlib import Path
import pasta


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Migrate to new versions of DeepgNN.")
    parser.add_argument(
        "--script_dir", type=str, required=True, help="Directory to migrate."
    )
    args = parser.parse_args()

    for filename in Path(args.script_dir).glob("**/*.py"):
        with open(filename, "r") as file:
            raw_input = file.read()

        tree = pasta.parse(raw_input)

        raw_output = pasta.dump(tree)

        raw_output = raw_output.replace("get_feature_type", "get_python_type")

        with open(filename, "w") as file:
            file.write(raw_output)
