# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
import json
import torch
import numpy as np
from deepgnn import TrainMode, setup_default_logging_config

from deepgnn.pytorch.modeling import BaseModel
from deepgnn.pytorch.common.ray_train import run_ray
from deepgnn.pytorch.common.dataset import TorchDeepGNNDataset
from deepgnn.graph_engine import GEEdgeSampler, GraphEngineBackend
from model import KGEModel  # type: ignore
from deepgnn import get_logger


def create_model(args: argparse.Namespace):
    get_logger().info(f"Creating KGEModel with seed:{args.seed}.")
    # set seed before instantiating the model

    model_args = json.loads(args.model_args)
    return KGEModel(
        num_negs=args.num_negs, gpu=args.cuda, embed_dim=args.dim, model_args=model_args
    )


def create_dataset(
    args: argparse.Namespace,
    model: BaseModel,
    rank: int = 0,
    world_size: int = 1,
    backend: GraphEngineBackend = None,
):
    return TorchDeepGNNDataset(
        sampler_class=GEEdgeSampler,
        backend=backend,
        query_fn=model.query if args.mode == TrainMode.TRAIN else model.query_eval,
        prefetch_queue_size=10,
        prefetch_worker_size=2,
        batch_size=args.batch_size,
        edge_types=np.array([args.edge_type], dtype=np.int32),
        epochs=1,
        sample_num=args.max_id // world_size,
        num_workers=world_size,
    )


def create_optimizer(args: argparse.Namespace, model: BaseModel, world_size: int):
    return torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.learning_rate * world_size,
        weight_decay=0,
    )


def _main():
    # setup default logging component.
    setup_default_logging_config(enable_telemetry=True)

    # run_dist is the unified entry for pytorch model distributed training/evaluation/inference.
    # User only needs to prepare initializing function for model, dataset, optimizer and args.
    # reference: `deepgnn/pytorch/training/factory.py`
    run_ray(
        init_model_fn=create_model,
        init_dataset_fn=create_dataset,
        init_optimizer_fn=create_optimizer,
    )


if __name__ == "__main__":
    _main()
