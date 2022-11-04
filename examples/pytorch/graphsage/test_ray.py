# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import pytest
import sys
import os
import platform
import tempfile
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

from deepgnn.pytorch.common.dataset import TorchDeepGNNDataset
from deepgnn.graph_engine import FileNodeSampler
from deepgnn.graph_engine.data.citation import Cora


feature_idx = 1
feature_dim = 50
label_idx = 0
label_dim = 121


def setup_module(module):
    import deepgnn.graph_engine.snark._lib as lib

    lib_name = "libwrapper.so"
    if platform.system() == "Windows":
        lib_name = "wrapper.dll"

    os.environ[lib._SNARK_LIB_PATH_ENV_KEY] = os.path.join(
        os.path.dirname(__file__), "..", "..", "..", "src", "cc", "lib", lib_name
    )


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
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
        logits = self.linear_relu_stack(x)
        return logits

    @staticmethod
    def query(g, idx):
        return {
            "features": g.node_features(
                idx, np.array([[feature_idx, feature_dim]]), np.float32
            ),
            "labels": np.ones((len(idx))),
        }


def onehot(values, size):
    values = values.squeeze()
    output = torch.zeros((values.shape[0], size))
    output[values.long()] = 1
    return output


def train_func(config: Dict):
    worker_batch_size = config["batch_size"]  # // session.get_world_size()

    model = NeuralNetwork()
    model = train.torch.prepare_model(model)

    optimizer = torch.optim.SGD(model.parameters(), lr=config["lr"])
    optimizer = train.torch.prepare_optimizer(optimizer)

    loss_fn = nn.CrossEntropyLoss()

    g = Client(config["data_dir"], [0])
    dataset = TorchDeepGNNDataset(
        sampler_class=FileNodeSampler,
        backend=g,  # type: ignore
        query_fn=NeuralNetwork.query,
        prefetch_queue_size=2,
        prefetch_worker_size=2,
        sample_files="/tmp/cora/train.nodes",
        batch_size=worker_batch_size,
        shuffle=True,
        drop_last=True,
        worker_index=0,
        num_workers=1,
    )
    dataset = torch.utils.data.DataLoader(
        dataset=dataset,
        num_workers=0,
    )

    model.train()
    for epoch in range(config["epochs"]):
        for i, batch in enumerate(dataset):
            pred = model(batch["features"])
            loss = loss_fn(pred.squeeze(), onehot(batch["labels"], 10))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return []


def test_graphsage_ppi_hvd_trainer():
    working_dir = tempfile.TemporaryDirectory()
    Cora(working_dir.name)

    ray.init()
    trainer = TorchTrainer(
        train_func,
        train_loop_config={"data_dir": working_dir.name, "lr": 1e-3, "batch_size": 64, "epochs": 4},
        scaling_config=ScalingConfig(num_workers=1, use_gpu=False),
    )
    result = trainer.fit()
    print(f"Results: {result.metrics}")
    working_dir.cleanup()


if __name__ == "__main__":
    sys.exit(
        pytest.main(
            [__file__, "--junitxml", os.environ["XML_OUTPUT_FILE"], *sys.argv[1:]]
        )
    )
