# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
import torch
import numpy as np
from deepgnn import TrainMode, setup_default_logging_config
from deepgnn import get_logger
from deepgnn.pytorch.common import F1Score
from deepgnn.pytorch.common.utils import get_python_type
from deepgnn.pytorch.encoding import get_feature_encoder
from deepgnn.pytorch.modeling import BaseModel
from deepgnn.graph_engine import (
    Graph,
    SamplingStrategy,
)
from model import PTGSupervisedGraphSage  # type: ignore
from typing import Dict, Callable
import os
import platform
import numpy as np
import torch
import ray
import ray.train as train
from ray.train.torch import TorchTrainer
from ray.air import session
from ray.air.config import ScalingConfig
from deepgnn import TrainMode, get_logger
from deepgnn.pytorch.common import get_args
from deepgnn.pytorch.common.utils import load_checkpoint, save_checkpoint
from deepgnn.graph_engine.snark.distributed import Server, Client as DistributedClient
from deepgnn.graph_engine.data.citation import Cora


# fmt: off
def init_args(parser: argparse.Namespace):
    group = parser.add_argument_group("GraphSAGE Parameters")
    group.add_argument("--algo", type=str, default="supervised", choices=["supervised"])
# fmt: on


def create_dataset(
    config: dict,
    model: BaseModel,
    rank: int = 0,
    world_size: int = 1,
    get_graph: Callable[..., DistributedClient] = None,  # type: ignore
) -> ray.data.DatasetPipeline:
    g = get_graph()
    max_id = g.node_count(0)
    dataset = ray.data.range(max_id).repartition(max_id // 140)
    pipe = dataset.window(blocks_per_window=4).repeat(config["num_epochs"])

    def transform_batch(idx: list):
        return model.query(g, np.array(idx))

    pipe = pipe.map_batches(transform_batch)
    return pipe


def train_func(config: Dict):
    """Training loop for ray trainer."""
    logger = get_logger()

    train.torch.accelerate()
    train.torch.enable_reproducibility(seed=session.get_world_rank())

    model = PTGSupervisedGraphSage(
        num_classes=config["label_dim"],
        metric=F1Score(),
        label_idx=config["label_idx"],
        label_dim=config["label_dim"],
        feature_dim=config["feature_dim"],
        feature_idx=config["feature_idx"],
        feature_type=np.float32,
        edge_type=0,
        fanouts=[5, 5],
        feature_enc=None,
    )
    model = train.torch.prepare_model(model)
    model.train()

    optimizer = torch.optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config["learning_rate"] * session.get_world_size(),
    )
    optimizer = train.torch.prepare_optimizer(optimizer)

    dataset = config["init_dataset_fn"](
        config,
        model,
        rank=session.get_world_rank(),
        world_size=session.get_world_size(),
        get_graph=config["get_graph"],
    )
    losses_full = []
    epoch_iter = (
        range(config["num_epochs"])
        if not hasattr(dataset, "iter_epochs")
        else dataset.iter_epochs()
    )
    for epoch, epoch_pipe in enumerate(epoch_iter):
        scores = []
        labels = []
        losses = []
        batch_iter = (
            dataset  # TODO reset every time
            if isinstance(epoch_pipe, int)
            else epoch_pipe.iter_torch_batches(batch_size=140)
        )
        for step, batch in enumerate(batch_iter):
            loss, score, label = model(batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            scores.append(score)
            labels.append(label)
            losses.append(loss.item())

        losses_full.extend(losses)
        steps_in_epoch_trained = 0

        if "model_path" in config:
            save_checkpoint(model, logger, epoch, step, model_dir=config["model_path"])

        session.report(
            {
                "metric": model.compute_metric(scores, labels).item(),
                "loss": np.mean(losses),
                "losses": losses_full,
            },
        )


def run_ray(init_dataset_fn, **kwargs):
    """Run ray trainer."""
    ray.init(num_cpus=4)

    cora = Cora()

    address = "localhost:9999"
    s = Server(address, cora.data_dir(), 0, 1)

    def get_graph():
        return DistributedClient([address])

    training_loop_config = {
        "get_graph": get_graph,
        "data_dir": cora.data_dir(),
        "num_epochs": 100,
        "feature_idx": 1,
        "feature_dim": 50,
        "label_idx": 0,
        "label_dim": 121,
        "num_classes": 7,
        "learning_rate": 0.005,
        "init_dataset_fn": init_dataset_fn,
        **kwargs,
    }
    if "training_args" in kwargs:
        training_loop_config.update(kwargs["training_args"])

    trainer = TorchTrainer(
        train_func,
        train_loop_config=training_loop_config,
        scaling_config=ScalingConfig(num_workers=1, use_gpu=False),
    )
    return trainer.fit()


def _main():
    # setup default logging component.
    setup_default_logging_config(enable_telemetry=True)

    # run_dist is the unified entry for pytorch model distributed training/evaluation/inference.
    # User only needs to prepare initializing function for model, dataset, optimizer and args.
    # reference: `deepgnn/pytorch/training/factory.py`
    run_ray(
        init_dataset_fn=create_dataset,
    )


if __name__ == "__main__":
    _main()
