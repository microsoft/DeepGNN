# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import sys
import os
import pytest
from factory import get_args
from omegaconf import DictConfig


def test_args():
    args = get_args(None, "conf")
    assert args.batch_size == 140

    args = get_args(
        None, DictConfig({"deepgnn": {"backend": "snark", "batch_size": 140}})
    )
    assert args.batch_size == 140


if __name__ == "__main__":
    sys.exit(
        pytest.main(
            [__file__, "--junitxml", os.environ["XML_OUTPUT_FILE"], *sys.argv[1:]]
        )
    )
