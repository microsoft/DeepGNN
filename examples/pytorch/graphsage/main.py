# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import torch
import numpy as np
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


def train_func(config: dict):
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

    g = DistributedClient(config["ge_address"])
    max_id = g.node_count(0)
    dataset = ray.data.range(max_id).repartition(max_id // config["batch_size"])
    pipe = dataset.window(blocks_per_window=4).repeat(config["num_epochs"])
    dataset = pipe.map_batches(lambda idx: model.query(g, np.array(idx)))

    for i, epoch in enumerate(dataset.iter_epochs()):
        scores = []
        labels = []
        losses = []
        for batch in epoch.iter_torch_batches(batch_size=config["batch_size"]):
            loss, score, label = model(batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            scores.append(score)
            labels.append(label)
            losses.append(loss.item())

        if i % 10 == 0:
            get_logger().info(
                f"Epoch {i:0>3d} {model.metric_name()}: {model.compute_metric(scores, labels).item():.4f} Loss: {np.mean(losses):.4f}"
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
    }

    TorchTrainer(
        train_func,
        train_loop_config=training_loop_config,
        scaling_config=ScalingConfig(num_workers=1, use_gpu=False),
        run_config=RunConfig(
            verbose=0,
        ),
    ).fit()


if __name__ == "__main__":
    _main()
