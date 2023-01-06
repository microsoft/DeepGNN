# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import numpy as np
import argparse
import torch

from deepgnn import str2list_int, setup_default_logging_config
from deepgnn import get_logger

from deepgnn.pytorch.modeling import BaseModel
from ray_util import run_ray

from model_geometric import GAT, GATQueryParameter  # type: ignore


# fmt: off
def init_args(parser):
    # GAT Model Parameters.
    parser.add_argument("--head_num", type=str2list_int, default="8,1", help="the number of attention headers.")
    parser.add_argument("--hidden_dim", type=int, default=8, help="hidden layer dimension.")
    parser.add_argument("--num_classes", type=int, default=-1, help="number of classes for category")
    parser.add_argument("--ffd_drop", type=float, default=0.0, help="feature dropout rate.")
    parser.add_argument("--attn_drop", type=float, default=0.0, help="attention layer dropout rate.")
    parser.add_argument("--l2_coef", type=float, default=0.0005, help="l2 loss")

    # GAT Query part
    parser.add_argument("--neighbor_edge_types", type=str2list_int, default="0", help="Graph Edge for attention encoder.",)

    # evaluate node file.
    parser.add_argument("--eval_file", default="", type=str, help="")
# fmt: on


def create_model(args: argparse.Namespace):
    get_logger().info(f"Creating GAT model with seed:{args.seed}.")
    # set seed before instantiating the model

    p = GATQueryParameter(
        neighbor_edge_types=np.array([args.neighbor_edge_types], np.int32),
        feature_idx=args.feature_idx,
        feature_dim=args.feature_dim,
        label_idx=args.label_idx,
        label_dim=args.label_dim,
    )

    return GAT(
        in_dim=args.feature_dim,
        head_num=args.head_num,
        hidden_dim=args.hidden_dim,
        num_classes=args.num_classes,
        ffd_drop=args.ffd_drop,
        attn_drop=args.attn_drop,
        q_param=p,
    )


def create_dataset(
    args: argparse.Namespace,
    model: BaseModel,
    rank: int = 0,
    world_size: int = 1,
    backend = None,
):
    return TorchDeepGNNDataset(
        sampler_class=FileNodeSampler,
        backend=backend,
        query_fn=model.q.query_training,
        prefetch_queue_size=2,
        prefetch_worker_size=2,
        sample_files=args.sample_file,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        worker_index=rank,
        num_workers=world_size,
    )


def create_eval_dataset(
    args: argparse.Namespace,
    model: BaseModel,
    rank: int = 0,
    world_size: int = 1,
    backend = None,
):
    return TorchDeepGNNDataset(
        sampler_class=FileNodeSampler,
        backend=backend,
        query_fn=model.q.query_training,
        prefetch_queue_size=2,
        prefetch_worker_size=2,
        sample_files=args.eval_file,
        batch_size=1000,
        shuffle=False,
        drop_last=True,
        worker_index=rank,
        num_workers=world_size,
    )


def create_optimizer(args: argparse.Namespace, model: BaseModel, world_size: int):
    return torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.learning_rate * world_size,
        weight_decay=0.0005,
    )


def _main():
    # setup default logging component.
    setup_default_logging_config(enable_telemetry=True)

    run_ray(
        init_model_fn=create_model,
        init_dataset_fn=create_dataset,
        init_optimizer_fn=create_optimizer,
        init_args_fn=init_args,
    )


if __name__ == "__main__":
    _main()
