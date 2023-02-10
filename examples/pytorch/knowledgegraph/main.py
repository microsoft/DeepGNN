# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
import json
import torch
import ray
import numpy as np
from deepgnn import TrainMode, setup_default_logging_config

from deepgnn.pytorch.modeling import BaseModel
from deepgnn.pytorch.common.ray_train import run_ray
from deepgnn.graph_engine import GEEdgeSampler, GraphEngineBackend
from model import KGEModel  # type: ignore
from deepgnn import get_logger
from deepgnn.graph_engine.snark.distributed import Client as DistributedClient


def create_model(args: argparse.Namespace):
    get_logger().info(f"Creating KGEModel with seed:{args.seed}.")
    # set seed before instantiating the model

    model_args = json.loads(args.model_args)
    return KGEModel(
        num_negs=args.num_negs, gpu=args.cuda, embed_dim=args.dim, model_args=model_args
    )


def create_dataset(
    args: argparse.Namespace,
    model: BaseModel,
    rank: int = 0,
    world_size: int = 1,
    address: str = "",
) -> ray.data.DatasetPipeline:
    g = DistributedClient([address])
    max_id = g.node_count(args.node_type) if args.max_id in [-1, None] else args.max_id
    dataset = ray.data.range(max_id).repartition(max_id // args.batch_size)
    pipe = dataset.window(blocks_per_window=4).repeat(args.num_epochs)

    def transform_batch(idx: list) -> dict:
        return model.q.query_training(g, np.array(idx))

    pipe = pipe.map_batches(transform_batch)
    return pipe


def create_optimizer(args: argparse.Namespace, model: BaseModel, world_size: int):
    return torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.learning_rate * world_size,
        weight_decay=0,
    )


def _main():
    # setup default logging component.
    setup_default_logging_config(enable_telemetry=True)

    # run_dist is the unified entry for pytorch model distributed training/evaluation/inference.
    # User only needs to prepare initializing function for model, dataset, optimizer and args.
    # reference: `deepgnn/pytorch/training/factory.py`
    ray.init(num_cpus=num_cpus)

    args = get_args(init_args_fn, kwargs["run_args"] if "run_args" in kwargs else None)

    trainer = TorchTrainer(
        train_func,
        train_loop_config={
            "args": args,
            "init_model_fn": create_model,
            "init_dataset_fn": create_dataset,
            "init_optimizer_fn": create_optimizer,
            **kwargs,
        },
        scaling_config=ScalingConfig(num_workers=1, use_gpu=args.gpu),
    )
    return trainer.fit()


if __name__ == "__main__":
    _main()
