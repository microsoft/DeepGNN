# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import numpy as np
import argparse
import torch
from torch.utils.data import DataLoader

from deepgnn import str2list_int, setup_default_logging_config
from deepgnn import get_logger
from deepgnn.pytorch.common.utils import set_seed
from deepgnn.pytorch.modeling import BaseModel
from deepgnn.pytorch.training import run_dist
from model import GAT, GATDataset, FileNodeSampler, BatchedSampler  # type: ignore


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
    if args.seed:
        set_seed(args.seed)

    return GAT(
        in_dim=args.feature_dim,
        head_num=args.head_num,
        hidden_dim=args.hidden_dim,
        num_classes=args.num_classes,
        ffd_drop=args.ffd_drop,
        attn_drop=args.attn_drop,
    )


def create_dataset(
    args: argparse.Namespace,
    model: BaseModel,
    rank: int = 0,
    world_size: int = 1,
):
    dataset = GATDataset(args.data_dir, [args.node_type], [args.feature_idx, args.feature_dim], [args.label_idx, args.label_dim], np.float32, np.float32)
    return DataLoader(
        dataset,
        sampler=BatchedSampler(FileNodeSampler(args.sample_file), args.batch_size),
        num_workers=2,
    )


def create_eval_dataset(
    args: argparse.Namespace,
    model: BaseModel,
    rank: int = 0,
    world_size: int = 1,
):
    dataset = GATDataset(args.data_dir, [args.node_type], [args.feature_idx, args.feature_dim], [args.label_idx, args.label_dim], np.float32, np.float32)
    return DataLoader(
        dataset,
        sampler=BatchedSampler(FileNodeSampler(args.eval_file), 1000),
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

    run_dist(
        init_model_fn=create_model,
        init_dataset_fn=create_dataset,
        init_optimizer_fn=create_optimizer,
        init_args_fn=init_args,
    )


if __name__ == "__main__":
    _main()
