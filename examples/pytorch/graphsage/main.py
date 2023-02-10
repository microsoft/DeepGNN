# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from typing import Dict
import torch
import numpy as np
import ray
import ray.train as train
from ray.train.torch import TorchTrainer
from ray.air import session
from ray.air.config import ScalingConfig, RunConfig

from deepgnn.graph_engine.snark.distributed import Server, Client as DistributedClient
from deepgnn.graph_engine.data.citation import Cora
from deepgnn import setup_default_logging_config, get_logger
from deepgnn.pytorch.common import F1Score
from deepgnn.pytorch.modeling import BaseModel
from model import PTGSupervisedGraphSage  # type: ignore
from deepgnn.pytorch.common.utils import save_checkpoint


def create_dataset(
    config: Dict,
    model: BaseModel,
) -> ray.data.DatasetPipeline:
    g = DistributedClient(config["ge_address"])
    max_id = g.node_count(0)
    dataset = ray.data.range(max_id).repartition(max_id // config["batch_size"])
    pipe = dataset.window(blocks_per_window=4).repeat(config["num_epochs"])
    return pipe.map_batches(lambda idx: model.query(g, np.array(idx)))


def train_func(config: Dict):
    """Training loop for ray trainer."""
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
            config["init_dataset_fn"](
                config,
                model,
            )
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
        if "model_path" in config:
            save_checkpoint(
                model, get_logger(), epoch, step, model_dir=config["model_path"]
            )

        session.report(
            {
                model.metric_name(): model.compute_metric(scores, labels).item(),
                "loss": np.mean(losses),
                "losses": losses_full,
            },
        )


def run_ray(init_dataset_fn, **kwargs):
    """Run ray trainer."""
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
        "init_dataset_fn": init_dataset_fn,
    }
    training_loop_config.update(kwargs["training_args"])

    trainer = TorchTrainer(
        train_func,
        train_loop_config=training_loop_config,
        scaling_config=ScalingConfig(num_workers=1, use_gpu=False),
    )
    return trainer.fit()


if __name__ == "__main__":
    run_ray(
        init_dataset_fn=create_dataset,
    )
