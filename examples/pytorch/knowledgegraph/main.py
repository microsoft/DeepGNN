# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
import json
import torch
import ray
import numpy as np
from deepgnn import TrainMode, setup_default_logging_config
import ray.train as train
from ray.train.torch import TorchTrainer
from ray.air import session
from ray.air.config import ScalingConfig, RunConfig

from deepgnn.pytorch.modeling import BaseModel
from deepgnn.pytorch.common.ray_train import run_ray
from deepgnn.graph_engine import GEEdgeSampler, GraphEngineBackend
from model import KGEModel  # type: ignore
from deepgnn import get_logger
from deepgnn.graph_engine.snark.distributed import Server, Client as DistributedClient
from deepgnn.graph_engine.data.citation import Cora


def train_func(config: dict):
    """Training loop for ray trainer."""
    train.torch.accelerate()
    train.torch.enable_reproducibility(seed=session.get_world_rank())

    address = "localhost:9999"
    cora = Cora()
    s = Server(address, cora.data_dir(), 0, 1)
    g = DistributedClient([address])

    with open("metadata.ini", "w") as f:
        f.write("[DEFAULT]")
        f.write("\n")
        f.write("num_entities=" + str(g.node_count(0)))
        f.write("\n")
        f.write("num_relations=" + str(g.edge_count(0)))

    model_args = json.loads(
        '{"double_entity_embedding":1,'
        ' "adversarial_temperature":1.0,'
        '"regularization":0.0, "gamma":24.0,'
        ' "metadata_path":"metadata.ini",'
        ' "score_func":"RotatE"}'
    )
    model = KGEModel(num_negs=10, embed_dim=1000, model_args=model_args)
    model = train.torch.prepare_model(model)
    model.train()

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=0.005,
        weight_decay=0,
    )
    optimizer = train.torch.prepare_optimizer(optimizer)

    max_id = g.node_count(0)
    dataset = ray.data.range(max_id).repartition(max_id // 140)
    pipe = dataset.window(blocks_per_window=4).repeat(10)
    pipe = pipe.map_batches(lambda idx: model.query(g, np.array(idx)))

    for i, epoch in enumerate(pipe.iter_epochs()):
        scores = []
        labels = []
        losses = []
        for batch in epoch.iter_torch_batches(batch_size=140):
            loss, score, label = model(batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            scores.append(score)
            labels.append(label)
            losses.append(loss.item())

        if i % 10 == 0:
            print(
                f"Epoch {i:0>3d} {model.metric_name()}: {model.compute_metric(scores, labels).item():.4f} Loss: {np.mean(losses):.4f}"
            )


def _main():
    # setup default logging component.
    setup_default_logging_config(enable_telemetry=True)

    # run_dist is the unified entry for pytorch model distributed training/evaluation/inference.
    # User only needs to prepare initializing function for model, dataset, optimizer and args.
    # reference: `deepgnn/pytorch/training/factory.py`
    ray.init(num_cpus=4)

    trainer = TorchTrainer(
        train_func,
        train_loop_config={},
        scaling_config=ScalingConfig(num_workers=1),
    )
    return trainer.fit()


if __name__ == "__main__":
    _main()
