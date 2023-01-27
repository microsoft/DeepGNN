# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
import torch
import ray
from deepgnn import TrainMode, setup_default_logging_config
from deepgnn import get_logger
from deepgnn.pytorch.common.utils import get_python_type
from deepgnn.pytorch.modeling import BaseModel
from args import init_args  # type: ignore
from model import HetGnnModel  # type: ignore
from sampler import HetGnnDataSampler  # type: ignore

from typing import Dict
import os
import platform
import numpy as np
import torch
import ray
import ray.train as train
from ray.train.torch import TorchTrainer
from ray.air import session
from ray.air.config import ScalingConfig
from ray.data import DatasetPipeline
from deepgnn import TrainMode, get_logger
from deepgnn.pytorch.common import get_args
from deepgnn.pytorch.common.utils import load_checkpoint, save_checkpoint
from deepgnn.graph_engine.snark.distributed import Server, Client as DistributedClient


def train_func(config: Dict):
    """Training loop for ray trainer."""
    args = config["args"]
    logger = get_logger()

    train.torch.accelerate(args.fp16)
    if args.seed:
        train.torch.enable_reproducibility(seed=args.seed + session.get_world_rank())

    model = HetGnnModel(
        node_type_count=args.node_type_count,
        neighbor_count=args.neighbor_count,
        embed_d=args.feature_dim,  # currently feature dimention is equal to embedding dimention.
        feature_type=get_python_type(args.feature_type),
        feature_idx=args.feature_idx,
        feature_dim=args.feature_dim,
    )
    model = train.torch.prepare_model(model, move_to_device=args.gpu)
    if args.mode == TrainMode.TRAIN:
        model.train()
    else:
        model.eval()

    epochs_trained, steps_in_epoch_trained = load_checkpoint(model, logger, args)

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.learning_rate * session.get_world_size(),
        weight_decay=0,
    )
    optimizer = train.torch.prepare_optimizer(optimizer)

    address = "localhost:9999"
    s = Server(address, args.data_dir, 0, len(args.partitions))
    g = DistributedClient([address])

    max_id = g.node_count(args.node_type) if args.max_id in [-1, None] else args.max_id

    for epoch in range(args.num_epochs):
        if epoch < epochs_trained:
            continue
        scores = []
        labels = []
        losses = []

        sampler = HetGnnDataSampler(g, max_id, args.batch_size, 3)

        for step, batch in enumerate(sampler):
            if step < steps_in_epoch_trained:
                continue
            loss, score, label = model(model.query(g, batch))
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


def run_ray(**kwargs):
    """Run ray trainer."""
    ray.init(num_cpus=4)

    args = get_args(init_args, kwargs["run_args"] if "run_args" in kwargs else None)

    trainer = TorchTrainer(
        train_func,
        train_loop_config={
            "args": args,
            **kwargs,
        },
        scaling_config=ScalingConfig(
            num_workers=1, use_gpu=args.gpu, resources_per_worker={"CPU": 2}
        ),
    )
    return trainer.fit()


def _main():
    # setup default logging component.
    setup_default_logging_config(enable_telemetry=True)

    # run_dist is the unified entry for pytorch model distributed training/evaluation/inference.
    # User only needs to prepare initializing function for model, dataset, optimizer and args.
    # reference: `deepgnn/pytorch/training/factory.py`
    ray.init(num_cpus=4)

    args = get_args(init_args)

    trainer = TorchTrainer(
        train_func,
        train_loop_config={
            "args": args,
            "init_model_fn": create_model,
            "init_dataset_fn": create_dataset,
            "init_optimizer_fn": create_optimizer,
        },
        scaling_config=ScalingConfig(num_workers=1, use_gpu=args.gpu),
    )
    return trainer.fit()


if __name__ == "__main__":
    _main()
