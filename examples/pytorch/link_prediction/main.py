# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from typing import Dict
import argparse
import os
import torch.optim
import ray
import numpy as np
from deepgnn import TrainMode, setup_default_logging_config
from deepgnn.pytorch.common.utils import (
    get_logger,
    get_python_type,
    get_store_name_and_path,
)
from deepgnn.pytorch.encoding import get_feature_encoder
from deepgnn.pytorch.modeling import BaseModel
from deepgnn.pytorch.common.ray_train import run_ray
from deepgnn.graph_engine import TextFileSampler, GraphEngineBackend
from args import init_args  # type: ignore
from consts import DEFAULT_VOCAB_CHAR_INDEX  # type: ignore
from model import LinkPredictionModel  # type: ignore
from deepgnn.graph_engine.snark.distributed import Client as DistributedClient
from ray.train.torch import TorchTrainer
from ray.air import session
from ray.air.config import ScalingConfig, RunConfig
import ray.train as train
from deepgnn.graph_engine.data.citation import Cora
from deepgnn.graph_engine.snark.distributed import Server, Client as DistributedClient
from deepgnn.pytorch.common import get_args


def train_func(config: Dict):
    """Training loop for ray trainer."""
    args = config["args"]

    train.torch.accelerate()
    train.torch.enable_reproducibility(seed=session.get_world_rank())

    feature_enc = get_feature_encoder(args)
    model = LinkPredictionModel(
        args=args,
        feature_dim=config["feature_dim"],
        feature_idx=config["feature_idx"],
        feature_type=np.float32,
        feature_enc=feature_enc[0],  # type: ignore
        vocab_index=feature_enc[1][DEFAULT_VOCAB_CHAR_INDEX],  # type: ignore
    )
    model = train.torch.prepare_model(model)
    model.train()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.005,
    )
    optimizer = train.torch.prepare_optimizer(optimizer)

    g = DistributedClient(config["ge_address"])
    max_id = g.node_count(0)
    dataset = ray.data.range(max_id).repartition(max_id // args.batch_size)
    pipe = dataset.window(blocks_per_window=4).repeat(args.num_epochs)
    pipe = pipe.map_batches(lambda idx: model.q.query_training(g, np.array(idx)))

    for epoch, epoch_pipe in enumerate(pipe.iter_epochs()):
        scores = []
        labels = []
        losses = []
        for step, batch in enumerate(
            epoch_pipe.iter_torch_batches(batch_size=args.batch_size)
        ):
            loss, score, label = model(batch)
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
    setup_default_logging_config(enable_telemetry=True)

    ray.init(num_cpus=4)

    address = "localhost:9999"
    cora = Cora()
    s = Server(address, cora.data_dir(), 0, 1)
    training_loop_config = {
        "ge_address": address,
        "batch_size": 140,
        "num_epochs": 200,
        "feature_idx": 1,
        "feature_dim": 50,
        "label_idx": 0,
        "label_dim": 121,
        "num_classes": 7,
        "learning_rate": 0.005,
        "args": get_args(),
    }

    trainer = TorchTrainer(
        train_func,
        train_loop_config=training_loop_config,
        scaling_config=ScalingConfig(num_workers=1),
    )
    return trainer.fit()


if __name__ == "__main__":
    _main()
