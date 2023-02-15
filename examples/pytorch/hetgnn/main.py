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
from deepgnn.graph_engine.data.citation import Cora


def train_func(config: Dict):
    """Training loop for ray trainer."""
    train.torch.accelerate()
    train.torch.enable_reproducibility(seed=session.get_world_rank())

    model = HetGnnModel(
        node_type_count=3,
        neighbor_count=5,
        embed_d=50,  # currently feature dimention is equal to embedding dimention.
        feature_type=np.float32,
        feature_idx=1,
        feature_dim=50,
    )
    model = train.torch.prepare_model(model)
    model.train()

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=0.005 * session.get_world_size(),
        weight_decay=0,
    )
    optimizer = train.torch.prepare_optimizer(optimizer)

    g = DistributedClient([config["ge_address"]])
    max_id = g.node_count(0)

    for epoch in range(10):
        scores = []
        labels = []
        losses = []

        sampler = HetGnnDataSampler(g, max_id, 140, 3)

        for step, batch in enumerate(sampler):
            loss, score, label = model(model.query(g, batch))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            scores.append(score)
            labels.append(label)
            losses.append(loss.item())

        session.report(
            {
                "metric": model.compute_metric(scores, labels).item(),
                "loss": np.mean(losses),
            },
        )


def _main():
    # setup default logging component.
    setup_default_logging_config(enable_telemetry=True)

    # run_dist is the unified entry for pytorch model distributed training/evaluation/inference.
    # User only needs to prepare initializing function for model, dataset, optimizer and args.
    # reference: `deepgnn/pytorch/training/factory.py`
    ray.init(num_cpus=4)

    cora = Cora()
    address = "localhost:9999"
    s = Server(address, cora.data_dir(), 0, 1)

    trainer = TorchTrainer(
        train_func,
        train_loop_config={
            "ge_address": address,
        },
        scaling_config=ScalingConfig(num_workers=1),
    )
    return trainer.fit()


if __name__ == "__main__":
    _main()
