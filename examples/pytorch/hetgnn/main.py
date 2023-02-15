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
from torch.utils.data import IterableDataset
from typing import Optional, Callable, Iterator
from inspect import signature
from deepgnn.graph_engine._base import Graph


class TorchDeepGNNDataset(IterableDataset):
    """Implementation of TorchDeepGNNDataset for use in a Torch Dataloader.
    TorchDeepGNNDataset initializes and executes a node or edge sampler given as
    sampler_class. For every batch of data requested, batch_size items are sampled
    from the sampler and passed to the given query_fn which pulls all necessaary
    information about the samples using the graph engine API. The output from
    the query function is passed to the trainer worker as the input to the
    model forward function.
    """

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
        sampler_class: HetGnnDataSampler,
        query_fn: Callable,
        backend=None,
        num_workers: int = 1,
        worker_index: int = 0,
        batch_size: int = 1,
        epochs: int = 1,
        enable_prefetch: bool = False,
        collate_fn: Optional[Callable] = None,
        # parameters to initialize samplers
        **kwargs,
    ):
        """Initialize DeepGNN dataset."""
        assert sampler_class is not None

        self.num_workers = num_workers
        self.sampler_class = sampler_class
        self.backend = backend
        self.worker_index = worker_index
        self.batch_size = batch_size
        self.epochs = epochs
        self.query_fn = query_fn
        self.enable_prefetch = enable_prefetch
        self.collate_fn = collate_fn
        self.kwargs = kwargs
        self.graph: DistributedClient = self.backend
        self.sampler = None

    def __iter__(self) -> Iterator:
        """Create an iterator for graph."""
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            self.kwargs.update(
                {
                    "data_parallel_index": worker_info.id,
                    "data_parallel_num": worker_info.num_workers,
                }
            )

        sig = signature(self.sampler_class.__init__)
        sampler_args = {}
        for key in sig.parameters:
            if key == "self":
                continue
            if sig.parameters[key].annotation == Graph:
                sampler_args[key] = self.graph
            elif key in self.kwargs.keys():
                sampler_args[key] = self.kwargs[key]
            elif hasattr(self, key):
                sampler_args[key] = getattr(self, key)

        self.sampler = self.sampler_class(**sampler_args)

        return self._DeepGNNDatasetIterator(
            graph=self.graph, sampler=self.sampler, query_fn=self.query_fn
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
        sampler_class=HetGnnDataSampler,
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
