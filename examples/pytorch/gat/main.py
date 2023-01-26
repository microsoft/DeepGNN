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
from deepgnn.pytorch.common.utils import load_checkpoint, save_checkpoint
from deepgnn.pytorch.modeling import BaseModel

from model_geometric import GAT, GATQueryParameter  # type: ignore
from deepgnn.graph_engine.snark.distributed import Server, Client as DistributedClient


def train_func(config: Dict):
    """Training loop for ray trainer."""
    logger = get_logger()
    model_dir = config["model_dir"]
    os.makedirs(model_dir, exist_ok=True)

    hvd.init()
    train.torch.enable_reproducibility(seed=session.get_world_rank())

    p = GATQueryParameter(
        neighbor_edge_types=np.array([0], np.int32),
        feature_idx=config["feature_idx"],
        feature_dim=config["feature_dim"],
        label_idx=config["label_idx"],
        label_dim=config["label_dim"],
    )
    model = GAT(
        in_dim=config["feature_dim"],
        head_num=[8, 1],
        hidden_dim=8,
        num_classes=config["num_classes"],
        ffd_drop=0.6,
        attn_drop=0.6,
        q_param=p,
    )

    model = train.torch.prepare_model(model)
    if config["mode"] == "train":
        model.train()
    else:
        model.eval()

    epochs_trained, steps_in_epoch_trained = load_checkpoint(
        model, logger, model_dir=model_dir, world_rank=session.get_world_rank()
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
    s = Server(address, config["data_dir"], 0, config["partitions"])
    g = DistributedClient([address])
    dataset = ray.data.read_text(config["sample_file"])
    dataset = dataset.repartition(dataset.count() // config["batch_size"])
    pipe = dataset.window(blocks_per_window=4).repeat(config["num_epochs"])

    def transform_batch(idx: list) -> dict:
        return model.q.query_training(g, np.array(idx))

    pipe = pipe.map_batches(transform_batch)

    for epoch, epoch_pipe in enumerate(pipe.iter_epochs()):
        if epoch < epochs_trained:
            continue
        scores = []
        labels = []
        losses = []
        for step, batch in enumerate(
            epoch_pipe.iter_torch_batches(batch_size=config["batch_size"])
        ):
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
        if epoch % 1 == 0:
            save_checkpoint(model, logger, epoch, step, model_dir=model_dir)

        session.report(
            {
                "metric": model.compute_metric(scores, labels).item(),
                "loss": np.mean(losses),
            },
        )


def _main():
    setup_default_logging_config(enable_telemetry=True)
    ray.init(num_cpus=4)

    trainer = HorovodTrainer(
        train_func,
        train_loop_config={
            "data_dir": "/tmp/cora",
            "model_dir": "/tmp/model_output",
            "sample_file": "/tmp/cora/train.nodes",
            "partitions": 1,
            "num_epochs": 180,
            "batch_size": 140,
            "feature_idx": 0,
            "feature_dim": 1433,
            "label_idx": 1,
            "label_dim": 1,
            "num_classes": 7,
            "mode": "train",
        },
        scaling_config=ScalingConfig(num_workers=1),
    )
    trainer.fit()

    trainer = HorovodTrainer(
        train_func,
        train_loop_config={
            "data_dir": "/tmp/cora",
            "model_dir": "/tmp/model_output",
            "sample_file": "/tmp/cora/test.nodes",
            "partitions": 1,
            "num_epochs": 1,
            "batch_size": 1000,
            "feature_idx": 0,
            "feature_dim": 1433,
            "label_idx": 1,
            "label_dim": 1,
            "num_classes": 7,
            "mode": "evaluate",
        },
        scaling_config=ScalingConfig(num_workers=1),
    )
    return trainer.fit()


if __name__ == "__main__":
    _main()
