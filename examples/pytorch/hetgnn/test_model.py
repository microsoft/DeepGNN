# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
import platform
import csv
import numpy as np
import numpy.testing as npt
import os
import sys
import pytest
import random
import tempfile
import time

from typing import Tuple

import torch
from torch.utils.data import IterableDataset

from deepgnn import get_logger
from deepgnn.pytorch.common.dataset import TorchDeepGNNDataset
from deepgnn.graph_engine import (
    Graph,
    SamplingStrategy,
    GraphType,
    BackendType,
    BackendOptions,
    GraphEngineBackend,
    create_backend,
)
from deepgnn.graph_engine.data.citation import Cora
from deepgnn.graph_engine.snark.converter.options import DataConverterType
from model import HetGnnModel  # type: ignore
from main import init_args, create_model, create_dataset, create_optimizer  # type: ignore
from ray_util import run_ray  # type: ignore
from sampler import HetGnnDataSampler  # type: ignore
import evaluation  # type: ignore

node_base_index = 1000000

logger = get_logger()


def setup_module(module):
    import deepgnn.graph_engine.snark._lib as lib

    lib_name = "libwrapper.so"
    if platform.system() == "Windows":
        lib_name = "wrapper.dll"

    os.environ[lib._SNARK_LIB_PATH_ENV_KEY] = os.path.join(
        os.path.dirname(__file__), "..", "..", "..", "src", "cc", "lib", lib_name
    )


def parse_testing_args(arg_str):
    parser = argparse.ArgumentParser(description="application data process")
    parser.add_argument("--A_n", type=int, default=28646, help="number of author node")
    parser.add_argument("--P_n", type=int, default=21044, help="number of paper node")
    parser.add_argument("--V_n", type=int, default=18, help="number of venue node")
    parser.add_argument("--C_n", type=int, default=4, help="number of node class label")
    parser.add_argument("--embed_d", type=int, default=128, help="embedding dimension")

    args = parser.parse_args(arg_str)
    return args


@pytest.fixture(scope="module")
def train_academic_data():
    model_dir = tempfile.TemporaryDirectory()
    working_dir = tempfile.TemporaryDirectory()
    Cora(working_dir.name)

    result = run_ray(
        init_model_fn=create_model,
        init_dataset_fn=create_dataset,
        init_optimizer_fn=create_optimizer,
        init_args_fn=init_args,
        run_args=[
            "--data_dir",
            working_dir.name,
            "--neighbor_count",
            "5",
            "--model_dir",
            model_dir.name,
            "--save_path",
            model_dir.name,
            "--num_epochs",
            "2",
            "--batch_size",
            "128",
            "--walk_length",
            "2",
            "--dim",
            "128",
            "--max_id",
            "1024",
            "--node_type_count",
            "3",
            "--feature_dim",
            "128",
            "--sample_file",
            os.path.join(working_dir.name, "academic", "a_node_list.txt"),
            "--feature_idx",
            "0",
        ],
    )

    yield os.path.join(model_dir.name, "gnnmodel-002-000000.pt"), result
    working_dir.cleanup()
    model_dir.cleanup()


def test_link_prediction_on_het_gnn(
    train_academic_data,  # noqa: F811
):
    model_dir, result = train_academic_data

    f1 = result.metrics["metric"]
    assert f1 > 0.6 and f1 < 0.9


if __name__ == "__main__":
    sys.exit(
        pytest.main(
            [__file__, "--junitxml", os.environ["XML_OUTPUT_FILE"], *sys.argv[1:]]
        )
    )
