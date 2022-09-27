# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

import ray

from deepgnn import TrainMode, setup_default_logging_config
from deepgnn import get_logger
from deepgnn.pytorch.common.utils import get_python_type, set_seed
from deepgnn.pytorch.modeling import BaseModel
from deepgnn.pytorch.training import run_dist
from args import init_args  # type: ignore
from sampler import HetGnnDataSampler  # type: ignore
from model import HetGnnModel, HetGNNDataset, FileNodeSampler, BatchedSampler  # type: ignore


def create_optimizer(args: argparse.Namespace, model: BaseModel, world_size: int):
    return torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config["learning_rate"] * world_size,
        weight_decay=0,
    )


import argparse
from typing import Dict
from ray.air import session

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

import ray.train as train
from ray.train.torch import TorchTrainer
from ray.air.config import ScalingConfig


def train_func(config: Dict):
    batch_size = config["batch_size"]
    lr = config["lr"]
    epochs = config["epochs"]

    worker_batch_size = batch_size // session.get_world_size()

    #get_logger().info(f"Creating HetGnnModel with seed:{config["seed}.")
    #set_seed(config["seed)

    model_original = HetGnnModel(
        node_type_count=1,#config["node_type_count,
        neighbor_count=5,#config["neighbor_count,
        embed_d=50,#config["feature_dim,  # currently feature dimention is equal to embedding dimention.
        feature_type=np.float32,#get_python_type(config["feature_type),
        feature_idx=1,#config["feature_idx,
        feature_dim=50,#config["feature_dim,
    )
    model = train.torch.prepare_model(model_original)

    if False:#config["mode == TrainMode.INFERENCE:
        dataset = HetGNNDataset(model_original.query_inference, config["data_dir"], [config["node_type"]], [config["feature_idx"], config["feature_dim"]], [config["label_idx"], config["label_dim"]], np.float32, np.float32)
        train_dataloader = DataLoader(dataset, sampler=FileNodeSampler(dataset.g, config["sample_file"]), batch_size=config["batch_size"])
    else:
        dataset = HetGNNDataset(model_original.query, config["data_dir"], [config["node_type"]], [config["feature_idx"], config["feature_dim"]], [config["label_idx"], config["label_dim"]], np.float32, np.float32)
        train_dataloader = DataLoader(dataset, sampler=HetGnnDataSampler(dataset.g, num_nodes=config["max_id"] // session.get_world_size(), batch_size=config["batch_size"], node_type_count=config["node_type_count"], walk_length=config["walk_length"]), batch_size=1)

    train_dataloader = train.torch.prepare_data_loader(train_dataloader)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    loss_results = []

    for _ in range(epochs):
        size = len(train_dataloader.dataset) // session.get_world_size()
        model.train()
        for batch, (X, y) in enumerate(train_dataloader):
            loss, score, label = model(X)
            #loss = loss_fn(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch % 100 == 0:
                loss, current = loss.item(), batch * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
        #session.report(dict(loss=loss))

    # return required for backwards compatibility with the old API
    # TODO(team-ml) clean up and remove return
    return loss_results


if __name__ == "__main__":
    import sys
    sys_args = sys.argv[1:]
    args = {sys_args[i * 2][2:]: int(sys_args[i * 2 + 1]) for i in range(len(sys_args) // 2)}
    args.update({
        "data_dir": "/tmp/cora",
        "mode": "train",
        "converter": "skip",
        "lr": 1e-3,
        "batch_size": 64,
        "epochs": 4,
        "node_type": 0,
        "walk_length": 2,
        "learning_rate": 0.005,
    })

    ray.init()
    trainer = TorchTrainer(
        train_func,
        train_loop_config=args,
        scaling_config=ScalingConfig(num_workers=1, use_gpu=False),
    )
    result = trainer.fit()
