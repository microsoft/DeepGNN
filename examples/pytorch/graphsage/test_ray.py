# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import pytest
import sys
import os
import platform
from typing import Dict

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, Sampler

import ray
import ray.train as train
from ray.train.torch import TorchTrainer
from ray.air import session
from ray.air.config import ScalingConfig

from deepgnn.graph_engine import SamplingStrategy
from deepgnn.graph_engine.snark.local import Client


import pandas as pd

feature_idx = 1
feature_dim = 50
label_idx = 0
label_dim = 121


class CoraDataset(Dataset):
    """Cora dataset with base torch sampler."""
    def __init__(self, node_types):
        self.g = Client("/tmp/cora", [0])
        self.node_types = np.array(node_types)
        self.count = self.g.node_count(self.node_types)

    def __len__(self):
        return self.count

    def __getitem__(self, idx):
        return {"features": self.g.node_features(idx, np.array([[feature_idx, feature_dim]]), feature_type=np.float32), "labels": np.ones((len(idx)))}

'''
class WeightedSampler(Sampler[int]):  # Shouldn't need this really with quick map from torch sampler?
    def __init__(self, graph, node_types):
        self.g = graph
        self.node_types = np.array(node_types)
        self.count = self.g.node_count(self.node_types)

    def __len__(self):
        return self.count

    def __iter__(self):
        for _ in range(len(self)):
            yield self.g.sample_nodes(1, self.node_types, SamplingStrategy.Weighted)[0]
'''


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


def train_epoch(dataloader, model, loss_fn, optimizer, size):
    model.train()

    for i, batch in enumerate(dataloader.random_shuffle_each_window().iter_torch_batches()):
        # Compute prediction error
        pred = model(batch["features"])
        loss = loss_fn(pred, batch["labels"].squeeze().long())

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            loss, current = loss.item(), i * len(batch["labels"])
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def train_func(config: Dict):
    batch_size = config["batch_size"]
    lr = config["lr"]
    epochs = config["epochs"]

    worker_batch_size = batch_size // session.get_world_size()

    """
    #ds = ray.data.range(2708, parallelism=1)
    #ds = ray.data.read_text("/tmp/cora/train.nodes", parallelism=1)
    from ray.data.datasource import SimpleTorchDatasource
    def generate_dataset():
        cora_dataset = CoraDataset([0])
        return cora_dataset.g.sample_nodes(2708, 0, SamplingStrategy.Random)

    ds = ray.data.read_datasource(
        SimpleTorchDatasource(), parallelism=1, dataset_factory=generate_dataset
    )

    def transform_batch(df: pd.DataFrame) -> pd.DataFrame:
        cora_dataset = CoraDataset([0])
        return cora_dataset.__getitem__(df)

    train_dataloader = ds.map_batches(transform_batch)
    """

    # pipeline incrementally on windows of the base data. This can be used for streaming data loading into ML training,
    # or to execute batch transformations on large datasets without needing to load the entire dataset into cluster memory.
    # Create a dataset and then create a pipeline from it.
    dataset = ray.data.range(2708, parallelism=2)

    print(dataset)
    # -> Dataset(num_blocks=200, num_rows=1000000, schema=<class 'int'>)

    # https://docs.ray.io/en/latest/data/pipelining-compute.html#pipelining-datasets
    # As a rule of thumb, higher parallelism settings perform better, however blocks_per_window == num_blocks effectively disables pipelining, since the DatasetPipeline will only contain a single Dataset.
    # The other extreme is setting blocks_per_window=1, which minimizes the latency to initial output but only allows one concurrent transformation task per stage:
    # As a rule of thumb, the cluster memory should be at least 2-5x the window size to avoid spilling.
    # TODO Check out the reported statistics for window size and blocks per window to ensure efficient pipeline execution.
    pipe = dataset.window(blocks_per_window=2)  # can be 10 or something
    print(pipe)
    # -> DatasetPipeline(num_windows=20, num_stages=1)

    # Applying transforms to pipelines adds more pipeline stages.
    def transform_batch(df: pd.DataFrame) -> pd.DataFrame:
        cora_dataset = CoraDataset([0])
        #assert False, df # ensure windows working
        return cora_dataset.__getitem__(df)
    pipe = pipe.map_batches(transform_batch)
    print(pipe)
    # -> DatasetPipeline(num_windows=20, num_stages=4)

    #train_dataloader = DataLoader(train_dataloader, batch_size=worker_batch_size)
    #train_dataloader = DataLoader(training_data, sampler=FileSampler("/tmp/cora/train.nodes"), batch_size=worker_batch_size)
    #train_dataloader = DataLoader(training_data, sampler=WeightedSampler(training_data.g, [0]), batch_size=worker_batch_size)

    #train_dataloader = train.torch.prepare_data_loader(train_dataloader)
    size = dataset.count() // session.get_world_size()  # TODO count affected by repeats?

    # Create model.
    model = NeuralNetwork()
    model = train.torch.prepare_model(model)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    loss_results = []

    for train_dataloader in pipe.repeat(epochs).iter_epochs():
        train_epoch(train_dataloader, model, loss_fn, optimizer, size)
        #loss = validate_epoch(test_dataloader, model, loss_fn)
        #loss_results.append(loss)
        #session.report(dict(loss=loss))

    return loss_results


def train_fashion_mnist(num_workers=2, use_gpu=False):
    trainer = TorchTrainer(
        train_func,
        train_loop_config={"lr": 1e-3, "batch_size": 64, "epochs": 4},
        scaling_config=ScalingConfig(num_workers=num_workers, use_gpu=use_gpu),
    )
    result = trainer.fit()
    print(f"Results: {result.metrics}")


def setup_module(module):
    import deepgnn.graph_engine.snark._lib as lib

    lib_name = "libwrapper.so"
    if platform.system() == "Windows":
        lib_name = "wrapper.dll"

    os.environ[lib._SNARK_LIB_PATH_ENV_KEY] = os.path.join(
        os.path.dirname(__file__), "..", "..", "..", "src", "cc", "lib", lib_name
    )


def test_graphsage_ppi_hvd_trainer():
    ray.init()
    train_fashion_mnist(num_workers=1, use_gpu=False)


if __name__ == "__main__":
    sys.exit(
        pytest.main(
            [__file__, "--junitxml", os.environ["XML_OUTPUT_FILE"], *sys.argv[1:]]
        )
    )
