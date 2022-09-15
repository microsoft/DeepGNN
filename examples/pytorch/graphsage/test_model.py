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
from torch.utils.data import Dataset, DataLoader

import ray
import ray.train as train
from ray.train.torch import TorchTrainer
from ray.air import session
from ray.air.config import ScalingConfig

import deepgnn.graph_engine.snark._lib as lib
from deepgnn.graph_engine.snark.local import Client


feature_idx = 1
feature_dim = 50
label_idx = 0
label_dim = 121

class CoraDataset(Dataset):
    def __init__(self):
        self.g = Client("/tmp/cora", [0, 1])

    def __len__(self):
        return self.g.node_count(np.array([0, 1, 2]))

    def __getitem__(self, idx):
        if isinstance(idx, (int, float)):
            idx = [idx]
        return self.g.node_features(idx, np.array([[feature_idx, feature_dim]]), feature_type=np.float32), torch.Tensor([0])


# Define model
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
    size = len(dataloader.dataset) // session.get_world_size()
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction error
        pred = model(X)
        print(pred.shape, y.shape, "OUTPUTTTT")
        loss = loss_fn(pred, y.squeeze().long())

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def validate_epoch(dataloader, model, loss_fn):
    size = len(dataloader.dataset) // session.get_world_size()
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(
        f"Test Error: \n "
        f"Accuracy: {(100 * correct):>0.1f}%, "
        f"Avg loss: {test_loss:>8f} \n"
    )
    return test_loss


def train_func(config: Dict):
    batch_size = config["batch_size"]
    lr = config["lr"]
    epochs = config["epochs"]

    worker_batch_size = batch_size // session.get_world_size()

    training_data = CoraDataset()

    # Create data loaders.
    train_dataloader = DataLoader(training_data, batch_size=worker_batch_size)
    #test_dataloader = DataLoader(test_data, batch_size=worker_batch_size)

    train_dataloader = train.torch.prepare_data_loader(train_dataloader)
    #test_dataloader = train.torch.prepare_data_loader(test_dataloader)

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
