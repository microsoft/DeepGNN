# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import sys
import platform
import pytest
import tempfile
import os
import shutil
import argparse
import torch
from deepgnn import TrainMode, setup_default_logging_config
from deepgnn import get_logger
from deepgnn.pytorch.common.utils import get_python_type
from deepgnn.pytorch.modeling import BaseModel
from args import init_args  # type: ignore
from model import HetGnnModel  # type: ignore
from sampler import HetGnnDataSampler  # type: ignore
from main import create_model, create_dataset, create_optimizer, init_args  # type: ignore
from deepgnn.graph_engine.data.ppi import PPI
from deepgnn.pytorch.common.ray_train import run_ray


def setup_module(module):
    import deepgnn.graph_engine.snark._lib as lib

    lib_name = "libwrapper.so"
    if platform.system() == "Windows":
        lib_name = "wrapper.dll"

    os.environ[lib._SNARK_LIB_PATH_ENV_KEY] = os.path.join(
        os.path.dirname(__file__), "..", "..", "..", "src", "cc", "lib", lib_name
    )


def test_run_args():
    # setup default logging component.
    setup_default_logging_config(enable_telemetry=True)

    data_dir = f"{tempfile.gettempdir()}/cora"
    PPI(data_dir)

    try:
        model_dir = f"{tempfile.gettempdir()}/hetgnn_test_args"
        shutil.rmtree(model_dir)
    except FileNotFoundError:
        pass

    run_args = f"--data_dir {data_dir} --mode train --seed 123 \
--backend snark --graph_type local --converter skip \
--batch_size 140 --learning_rate 0.005 --num_epochs 10 --max_id -1 --node_type_count 3 \
--model_dir {model_dir} --metric_dir {model_dir} --save_path {model_dir} --max_id 140 \
--feature_idx 1 --feature_dim 50 --label_idx 0 --label_dim 121 \
--log_by_steps 1 --data_parallel_num 0".split()

    # run_dist is the unified entry for pytorch model distributed training/evaluation/inference.
    # User only needs to prepare initializing function for model, dataset, optimizer and args.
    # reference: `deepgnn/pytorch/training/factory.py`
    result = run_ray(
        init_model_fn=create_model,
        init_dataset_fn=create_dataset,
        init_optimizer_fn=create_optimizer,
        init_args_fn=init_args,
        num_cpus=4,
        run_args=run_args,
    )


if __name__ == "__main__":
    sys.exit(
        pytest.main(
            [__file__, "--junitxml", os.environ["XML_OUTPUT_FILE"], *sys.argv[1:]]
        )
    )
