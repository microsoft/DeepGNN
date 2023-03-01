# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Optional, Callable, Iterator
import argparse
from typing import Dict
import os
import platform
import numpy as np
import torch
from torch.utils.data import IterableDataset
import ray
import ray.train as train
from ray.train.torch import TorchTrainer
from ray.air import session
from ray.air.config import ScalingConfig
from deepgnn.graph_engine.snark.distributed import Server, Client as DistributedClient
from deepgnn.graph_engine.data.citation import Cora
from args import init_args  # type: ignore
from model import HetGnnModel  # type: ignore
from sampler import HetGnnDataSampler  # type: ignore


class TorchDeepGNNDataset(IterableDataset):
    """Implementation of TorchDeepGNNDataset for use in a Torch Dataloader."""

    class _DeepGNNDatasetIterator:
        def __init__(
            self,
            graph: DistributedClient,
            sampler: HetGnnDataSampler,
            query_fn: Callable,
        ):
            self.graph = graph
            self.sampler = sampler
            self.query_fn = query_fn
            self.sampler_iter = iter(self.sampler)

        def __next__(self):
            inputs = next(self.sampler_iter)
            graph_tensor = self.query_fn(self.graph, inputs)
            return graph_tensor

        def __iter__(self):
            return self

    def __init__(
        self,
        query_fn: Callable,
        backend=None,
        num_nodes: int = -1,
        batch_size: int = 1,
        node_type_count: int = -1,
        walk_length: int = -1,
    ):
        """Initialize DeepGNN dataset."""
        self.graph: DistributedClient = backend
        self.query_fn = query_fn

        self.num_nodes = batch_size
        self.batch_size = batch_size
        self.node_type_count = node_type_count
        self.walk_length = walk_length

    def __iter__(self) -> Iterator:
        """Create an iterator for graph."""
        sampler = HetGnnDataSampler(
            self.graph,
            self.num_nodes,
            self.batch_size,
            self.node_type_count,
            self.walk_length,
        )

        return self._DeepGNNDatasetIterator(
            graph=self.graph, sampler=sampler, query_fn=self.query_fn
        )


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

    dataset = TorchDeepGNNDataset(
        backend=g,
        query_fn=model.query,
        num_nodes=max_id,
        batch_size=140,
        node_type_count=1,
        walk_length=3,
    )

    for epoch in range(10):
        scores = []
        labels = []
        losses = []

        for step, batch in enumerate(dataset):
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
