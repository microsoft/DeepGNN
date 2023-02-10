# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Optional, Dict
import os
import platform
import numpy as np
import argparse
import torch

import ray
import ray.train as train
import horovod.torch as hvd
import ray.train.torch
from ray.train.horovod import HorovodTrainer
from ray.air import session
from ray.air.config import ScalingConfig

from deepgnn import str2list_int, setup_default_logging_config
from deepgnn import get_logger
from deepgnn import TrainMode, get_logger
from deepgnn.graph_engine import create_backend, BackendOptions
from deepgnn.graph_engine.samplers import GENodeSampler, GEEdgeSampler
from deepgnn.pytorch.common import get_args
from deepgnn.pytorch.common.utils import load_checkpoint, save_checkpoint
from deepgnn.pytorch.modeling import BaseModel

from model_geometric import GAT, GATQueryParameter  # type: ignore
from deepgnn.graph_engine.snark.distributed import Server, Client as DistributedClient


# fmt: off
def init_args(parser):
    # GAT Model Parameters.
    parser.add_argument("--head_num", type=str2list_int, default="8,1", help="the number of attention headers.")
    parser.add_argument("--hidden_dim", type=int, default=8, help="hidden layer dimension.")
    parser.add_argument("--num_classes", type=int, default=-1, help="number of classes for category")
    parser.add_argument("--ffd_drop", type=float, default=0.6, help="feature dropout rate.")
    parser.add_argument("--attn_drop", type=float, default=0.6, help="attention layer dropout rate.")
    parser.add_argument("--l2_coef", type=float, default=0.0005, help="l2 loss")

    # GAT Query part
    parser.add_argument("--neighbor_edge_types", type=str2list_int, default="0", help="Graph Edge for attention encoder.",)
# fmt: on


def train_func(config: Dict):
    """Training loop for ray trainer."""
    args = config["args"]

    logger = get_logger()
    os.makedirs(args.save_path, exist_ok=True)

    hvd.init()
    if args.seed:
        train.torch.enable_reproducibility(seed=args.seed + session.get_world_rank())

    p = GATQueryParameter(
        neighbor_edge_types=np.array([args.neighbor_edge_types], np.int32),
        feature_idx=args.feature_idx,
        feature_dim=args.feature_dim,
        label_idx=args.label_idx,
        label_dim=args.label_dim,
    )
    model = GAT(
        in_dim=args.feature_dim,
        head_num=args.head_num,
        hidden_dim=args.hidden_dim,
        num_classes=args.num_classes,
        ffd_drop=args.ffd_drop,
        attn_drop=args.attn_drop,
        q_param=p,
    )

    # https://docs.ray.io/en/latest/tune/api_docs/trainable.html#function-api-checkpointing
    model = train.torch.prepare_model(model, move_to_device=args.gpu)
    if args.mode == TrainMode.TRAIN:
        model.train()
    else:
        model.eval()

    epochs_trained, steps_in_epoch_trained = load_checkpoint(
        model, logger, args, session.get_world_rank()
    )

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=0.005 * session.get_world_size(),
        weight_decay=0.0005,
    )
    optimizer = hvd.DistributedOptimizer(
        optimizer, named_parameters=model.named_parameters()
    )

    address = "localhost:9999"
    s = Server(address, args.data_dir, 0, len(args.partitions))
    g = DistributedClient([address])
    max_id = g.node_count(args.node_type) if args.max_id in [-1, None] else args.max_id
    dataset = ray.data.range(max_id).repartition(max_id // args.batch_size)
    pipe = dataset.window(blocks_per_window=4).repeat(args.num_epochs)

    def transform_batch(idx: list) -> dict:
        return model.q.query_training(g, np.array(idx))

    pipe = pipe.map_batches(transform_batch)

    for epoch in range(epochs_trained, args.num_epochs):
        scores = []
        labels = []
        losses = []
        for step, batch in enumerate(dataset):
            if step < steps_in_epoch_trained:
                continue
            loss, score, label = model(batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            scores.append(score)
            labels.append(label)
            losses.append(loss.item())

        steps_in_epoch_trained = 0
        if epoch % args.save_ckpt_by_epochs == 0:
            save_checkpoint(model, logger, epoch, step, args)

        session.report(
            {
                "metric": model.compute_metric(scores, labels).item(),
                "loss": np.mean(losses),
            },
        )


def _main():
    # setup default logging component.
    setup_default_logging_config(enable_telemetry=True)

    ray.init(num_cpus=4)

    args = get_args(init_args)

    trainer = HorovodTrainer(
        train_func,
        train_loop_config={"args": args},
        scaling_config=ScalingConfig(num_workers=1, use_gpu=args.gpu),
    )
    return trainer.fit()


if __name__ == "__main__":
    _main()
