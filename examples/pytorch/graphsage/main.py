# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
import torch
import numpy as np
import os
import sys
import random
from deepgnn import TrainMode, setup_default_logging_config
from deepgnn import get_logger
from deepgnn.pytorch.common import MRR, F1Score
from deepgnn.pytorch.common.utils import get_feature_type, set_seed
from deepgnn.pytorch.encoding import get_feature_encoder
from deepgnn.pytorch.modeling import BaseModel
from deepgnn.pytorch.training import run_dist
from deepgnn.pytorch.common.dataset import TorchDeepGNNDataset
from deepgnn.graph_engine import (
    CSVNodeSampler,
    GENodeSampler,
    SamplingStrategy,
    GraphEngineBackend,
)
from model import SupervisedGraphSage, UnSupervisedGraphSage


# fmt: off
def init_args(parser: argparse.Namespace):
    group = parser.add_argument_group("GraphSAGE Parameters")
    group.add_argument("--algo", type=str, default="unsupervised", choices=["unsupervised", "supervised"])
# fmt: on


def create_model(args: argparse.Namespace):

    # set seed before instantiating the model
    if args.seed:
        set_seed(args.seed)

    feature_enc = get_feature_encoder(args)

    if args.algo == "supervised":
        get_logger().info(f"Creating SupervisedGraphSage model with seed:{args.seed}.")
        return SupervisedGraphSage(
            num_classes=args.label_dim,
            metric=F1Score(),
            label_idx=args.label_idx,
            label_dim=args.label_dim,
            feature_dim=args.feature_dim,
            feature_idx=args.feature_idx,
            feature_type=get_feature_type(args.feature_type),
            edge_type=args.node_type,
            fanouts=args.fanouts,
            feature_enc=feature_enc,
        )

    elif args.algo == "unsupervised":
        get_logger().info(
            f"Creating UnSupervisedGraphSage model with seed:{args.seed}."
        )
        return UnSupervisedGraphSage(
            num_classes=args.dim,
            metric=MRR(),
            num_negs=args.num_negs,
            feature_dim=args.feature_dim,
            feature_idx=args.feature_idx,
            feature_type=get_feature_type(args.feature_type),
            edge_type=args.node_type,
            fanouts=args.fanouts,
            feature_enc=feature_enc,
            embed_dim=args.dim,
        )
    else:
        raise RuntimeError(f"Unknown algo: {args.algo}")


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
            num_workers=world_size,
            worker_index=rank,
            batch_size=args.batch_size,
            sample_file=args.sample_file,
            query_fn=model.query,
            prefetch_queue_size=10,
            prefetch_worker_size=2,
        )
    else:
        return TorchDeepGNNDataset(
            sampler_class=GENodeSampler,
            backend=backend,
            sample_num=args.max_id,
            num_workers=world_size,
            worker_index=rank,
            node_types=np.array([args.node_type], dtype=np.int32),
            batch_size=args.batch_size,
            query_fn=model.query,
            prefetch_queue_size=10,
            prefetch_worker_size=2,
            strategy=SamplingStrategy.RandomWithoutReplacement,
        )


def create_optimizer(args: argparse.Namespace, model: BaseModel, world_size: int):
    return torch.optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.learning_rate * world_size,
    )


def _main():
    run_args = None
    if len(sys.argv) >= 3 and sys.argv[1] == "--conf-dir":
        run_args = f"{os.path.dirname(os.path.abspath(__file__))}/{sys.argv[2]}"

    # setup default logging component.
    setup_default_logging_config(enable_telemetry=True)

    random.seed(42)
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
    _main()
