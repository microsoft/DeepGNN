# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Dict
import os
import tempfile
import numpy as np
import torch
import ray
import ray.train as train
import horovod.torch as hvd
import ray.train.torch
from ray.train.horovod import HorovodTrainer
from ray.air import session
from ray.air.config import ScalingConfig

from deepgnn import setup_default_logging_config
from model_geometric import GAT, GATQueryParameter  # type: ignore
from deepgnn.graph_engine.snark.distributed import Server, Client as DistributedClient
from deepgnn.graph_engine.data.citation import Cora


def train_func(config: Dict):
    """Training loop for ray trainer."""
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

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=0.005 * session.get_world_size(),
        weight_decay=0.0005,
    )
    optimizer = hvd.DistributedOptimizer(
        optimizer, named_parameters=model.named_parameters()
    )

    g = config["get_graph"]()
    batch_size = 140
    train_dataset = ray.data.read_text(f"{config['data_dir']}/train.nodes")
    train_dataset = train_dataset.repartition(train_dataset.count() // batch_size)
    train_pipe = train_dataset.window(blocks_per_window=4).repeat(config["num_epochs"])

    def transform_batch(idx: list) -> dict:
        return model.q.query_training(g, np.array(idx))

    train_pipe = train_pipe.map_batches(transform_batch)

    test_dataset = ray.data.read_text(f"{config['data_dir']}/test.nodes")
    test_dataset = test_dataset.repartition(1)
    test_dataset = test_dataset.map_batches(transform_batch)
    test_dataset_iter = test_dataset.repeat(config["num_epochs"]).iter_epochs()

    for epoch, epoch_pipe in enumerate(train_pipe.iter_epochs()):
        model.train()
        losses = []
        for step, batch in enumerate(
            epoch_pipe.iter_torch_batches(batch_size=batch_size)
        ):
            loss, score, label = model(batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        model.eval()
        batch = next(next(test_dataset_iter).iter_torch_batches(batch_size=1000))
        loss, score, label = model(batch)
        test_scores = [score]
        test_labels = [label]

        session.report(
            {
                "test_metric": model.compute_metric(test_scores, test_labels).item(),
                "loss": np.mean(losses),
            },
        )


def _main():
    setup_default_logging_config(enable_telemetry=True)
    ray.init(num_cpus=4)

    data_dir = tempfile.TemporaryDirectory()
    Cora(data_dir.name)

    address = "localhost:9999"
    s = Server(address, data_dir.name, 0, 1)

    def get_graph():
        return DistributedClient([address])

    trainer = HorovodTrainer(
        train_func,
        train_loop_config={
            "get_graph": get_graph,
            "data_dir": data_dir.name,
            "num_epochs": 180,
            "feature_idx": 0,
            "feature_dim": 1433,
            "label_idx": 1,
            "label_dim": 1,
            "num_classes": 7,
            "mode": "train",
        },
        scaling_config=ScalingConfig(num_workers=1),
    )
    return trainer.fit()


if __name__ == "__main__":
    _main()
