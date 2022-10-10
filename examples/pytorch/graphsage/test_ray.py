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
        self.g = Client("/tmp/cora", [0, 1])
        self.node_types = np.array(node_types)
        self.count = self.g.node_count(self.node_types)

    def __len__(self):
        return self.count

    def __getitem__(self, idx):
        return {"features": self.g.node_features(idx, np.array([[feature_idx, feature_dim]]), feature_type=np.float32), "labels": np.ones((len(idx)))}

# Range sampling use this w/ SubsetRandomSampler w/ list(range(start, stop))

'''
class CoraDataset(Dataset):
    """Cora dataset with file sampler."""
    def __init__(self, node_types):
        self.g = Client("/tmp/cora", [0, 1])  # TODO utility function for adl
        self.node_types = np.array(node_types)
        self.count = self.g.node_count(self.node_types)

    def __len__(self):
        return self.count

    def __getitem__(self, sampler_idx):
        if isinstance(sampler_idx, (int, float)):
            sampler_idx = [sampler_idx]
        idx = sampler_idx
        return self.g.node_features(idx, np.array([[feature_idx, feature_dim]]), feature_type=np.float32), torch.Tensor([0])


class FileSampler(Sampler[int]):  # Shouldn't need this really with quick map from torch sampler?
    def __init__(self, filename):
        self.filename = filename

    def __len__(self) -> int:
        raise NotImplementedError("")

    def __iter__(self):
        with open(self.filename, "r") as file:
            for line in file.readlines():
                yield int(line)
'''
'''
class CoraDataset(Dataset):
    """Cora dataset with file sampler."""
    def __init__(self, node_types):
        self.g = Client("/tmp/cora", [0])
        self.node_types = np.array(node_types)
        self.count = self.g.node_count(self.node_types)

    def __len__(self):
        return self.count

    def __getitem__(self, sampler_idx):
        if isinstance(sampler_idx, (int, float)):
            sampler_idx = [sampler_idx]
        idx = sampler_idx
        return self.g.node_features(idx, np.array([[feature_idx, feature_dim]]), feature_type=np.float32), torch.Tensor([0])


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


def train_epoch(dataloader, model, loss_fn, optimizer):
    size = dataloader.count() // session.get_world_size()
    model.train()
    for i, batch in enumerate(dataloader.iter_torch_batches()):
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

    #training_data = CoraDataset([0])

    ds = ray.data.range(2708)
    def transform_batch(df: pd.DataFrame) -> pd.DataFrame:
        cora_dataset = CoraDataset([0])
        return cora_dataset.__getitem__(df)
    train_dataloader = ds.map_batches(transform_batch)

    #print(ds.show())
    #train_dataloader = DataLoader(train_dataloader, batch_size=worker_batch_size)
    #train_dataloader = DataLoader(training_data, sampler=FileSampler("/tmp/cora/train.nodes"), batch_size=worker_batch_size)
    #train_dataloader = DataLoader(training_data, sampler=WeightedSampler(training_data.g, [0]), batch_size=worker_batch_size)

    #train_dataloader = train.torch.prepare_data_loader(train_dataloader)

    # Create model.
    model = NeuralNetwork()
    model = train.torch.prepare_model(model)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    loss_results = []

    for _ in range(epochs):
        train_epoch(train_dataloader, model, loss_fn, optimizer)
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
