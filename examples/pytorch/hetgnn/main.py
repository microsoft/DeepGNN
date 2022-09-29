# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Dict

import numpy as np

import torch
from torch import nn
from torch.utils.data import DataLoader

import ray
import ray.train as train
from ray.train.torch import TorchTrainer
from ray.air import session
from ray.air.config import ScalingConfig, RunConfig

from deepgnn import setup_default_logging_config
from deepgnn import get_logger
from deepgnn.pytorch.common.utils import get_python_type, set_seed

from args import init_args  # type: ignore
from sampler import HetGnnDataSampler, FileNodeSampler, BatchedSampler  # type: ignore
from model import HetGnnModel, HetGNNDataset  # type: ignore


def train_func(config: Dict):
    import os
    import platform
    import deepgnn.graph_engine.snark._lib as lib

    lib_name = "libwrapper.so"
    if platform.system() == "Windows":
        lib_name = "wrapper.dll"

    os.environ[lib._SNARK_LIB_PATH_ENV_KEY] = os.path.join(
        os.path.dirname(__file__), "..", "..", "..", "src", "cc", "lib", lib_name
    )

    batch_size = config["batch_size"]
    epochs = config["num_epochs"]
    world_size = session.get_world_size()

    worker_batch_size = batch_size // world_size
    num_nodes = config["max_id"] // world_size
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
        train_dataloader = DataLoader(dataset, sampler=HetGnnDataSampler(dataset.g, num_nodes=num_nodes, batch_size=config["batch_size"], node_type_count=config["node_type_count"], walk_length=config["walk_length"], sample_files=config["sample_file"]), batch_size=1)

    train_dataloader = train.torch.prepare_data_loader(train_dataloader)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config["learning_rate"] * world_size,
        weight_decay=0,
    )
    loss_results = []

    model.train()
    for _ in range(epochs):
        for batch, (X, y) in enumerate(train_dataloader):
            loss, score, label = model(X)
            #loss = loss_fn(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch % 100 == 0:
                loss, current = loss.item(), batch * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{num_nodes:>5d}]")
        #session.report(dict(loss=loss))

    # return required for backwards compatibility with the old API
    # TODO(team-ml) clean up and remove return
    return loss_results


if __name__ == "__main__":
    from deepgnn.pytorch.training.args import get_args
    args = get_args(init_args, run_args=None)

    ray.init()
    trainer = TorchTrainer(
        train_func,
        train_loop_config=vars(args),
        run_config=RunConfig(verbose=1),
        scaling_config=ScalingConfig(num_workers=1, use_gpu=False),
    )
    result = trainer.fit()
