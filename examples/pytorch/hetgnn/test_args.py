# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import sys
import pytest
import os
import shutil
import argparse
import torch
from deepgnn import TrainMode, setup_default_logging_config
from deepgnn import get_logger
from deepgnn.pytorch.common.utils import get_dtype, set_seed
from deepgnn.pytorch.modeling import BaseModel
from deepgnn.pytorch.training import run_dist
from deepgnn.pytorch.common.dataset import TorchDeepGNNDataset
from deepgnn.graph_engine import CSVNodeSampler, GraphEngineBackend
from args import init_args
from model import HetGnnModel
from sampler import HetGnnDataSampler
from deepgnn.graph_engine.data.ppi import PPI


def create_model(args: argparse.Namespace):
    get_logger().info(f"Creating HetGnnModel with seed:{args.seed}.")
    # set seed before instantiating the model
    if args.seed:
        set_seed(args.seed)

    return HetGnnModel(
        node_type_count=args.node_type_count,
        neighbor_count=args.neighbor_count,
        embed_d=args.feature_dim,  # currently feature dimention is equal to embedding dimention.
        dtype=get_dtype(args.dtype),
        feature_idx=args.feature_idx,
        feature_dim=args.feature_dim,
    )


def create_dataset(
    args: argparse.Namespace,
    model: BaseModel,
    rank: int = 0,
    world_size: int = 1,
    backend: GraphEngineBackend = None,
):
    if args.mode == TrainMode.INFERENCE:
        return TorchDeepGNNDataset(
            sampler_class=CSVNodeSampler,
            backend=backend,
            query_fn=model.query_inference,
            prefetch_queue_size=10,
            prefetch_worker_size=2,
            batch_size=args.batch_size,
            sample_file=args.sample_file,
        )
    else:
        return TorchDeepGNNDataset(
            sampler_class=HetGnnDataSampler,
            backend=backend,
            query_fn=model.query,
            prefetch_queue_size=10,
            prefetch_worker_size=2,
            num_nodes=args.max_id // world_size,
            batch_size=args.batch_size,
            node_type_count=args.node_type_count,
            walk_length=args.walk_length,
        )


def create_optimizer(args: argparse.Namespace, model: BaseModel, world_size: int):
    return torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.learning_rate * world_size,
        weight_decay=0,
    )


def test_run_args():
    # setup default logging component.
    setup_default_logging_config(enable_telemetry=True)

    data_dir = "/tmp/cora"
    PPI(data_dir)

    try:
        model_dir = "/tmp/hetgnn_test_args"
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
    run_dist(
        init_model_fn=create_model,
        init_dataset_fn=create_dataset,
        init_optimizer_fn=create_optimizer,
        init_args_fn=init_args,
        run_args=run_args,
    )


if __name__ == "__main__":
    sys.exit(
        pytest.main(
            [__file__, "--junitxml", os.environ["XML_OUTPUT_FILE"], *sys.argv[1:]]
        )
    )
