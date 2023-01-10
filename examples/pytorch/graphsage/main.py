# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
import torch
import numpy as np
from deepgnn import TrainMode, setup_default_logging_config
from deepgnn import get_logger
from deepgnn.pytorch.common import F1Score
from deepgnn.pytorch.common.utils import get_python_type
from deepgnn.pytorch.encoding import get_feature_encoder
from deepgnn.pytorch.modeling import BaseModel
from deepgnn.graph_engine import (
    Graph,
    SamplingStrategy,
)
from model import PTGSupervisedGraphSage  # type: ignore
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
from deepgnn import TrainMode, get_logger
from deepgnn.pytorch.common import get_args
from deepgnn.pytorch.common.utils import load_checkpoint, save_checkpoint
from deepgnn.graph_engine.snark.distributed import Server, Client as DistributedClient


# fmt: off
def init_args(parser: argparse.Namespace):
    group = parser.add_argument_group("GraphSAGE Parameters")
    group.add_argument("--algo", type=str, default="supervised", choices=["supervised"])
# fmt: on


def create_dataset(
    args: argparse.Namespace,
    model: BaseModel,
    rank: int = 0,
    world_size: int = 1,
    backend=None,
):
    address = "localhost:9999"
    g = DistributedClient([address])
    # NOTE: See https://deepgnn.readthedocs.io/en/latest/graph_engine/dataset.html
    #       for how to use a different sampler
    max_id = g.node_count(args.node_type) if args.max_id in [-1, None] else args.max_id
    dataset = ray.data.range(max_id).repartition(max_id // args.batch_size)
    pipe = dataset.window(blocks_per_window=4).repeat(args.num_epochs)

    def transform_batch(idx: list) -> dict:
        # If get Ray error with return shape, use deepgnn.graph_engine.util.serialize/deserialize
        # in your query and forward function
        return model.query(g, np.array(idx))  # TODO Update to your query function

    pipe = pipe.map_batches(transform_batch)
    return pipe


def train_func(config: Dict):
    """Training loop for ray trainer."""
    args = config["args"]

    logger = get_logger()
    os.makedirs(args.save_path, exist_ok=True)

    train.torch.accelerate(args.fp16)
    if args.seed:
        train.torch.enable_reproducibility(seed=args.seed + session.get_world_rank())

    feature_enc = get_feature_encoder(args)
    model = PTGSupervisedGraphSage(
        num_classes=args.label_dim,
        metric=F1Score(),
        label_idx=args.label_idx,
        label_dim=args.label_dim,
        feature_dim=args.feature_dim,
        feature_idx=args.feature_idx,
        feature_type=get_python_type(args.feature_type),
        edge_type=args.node_type,
        fanouts=args.fanouts,
        feature_enc=feature_enc,
    )
    model = train.torch.prepare_model(model, move_to_device=args.gpu)
    if args.mode == TrainMode.TRAIN:
        model.train()
    else:
        model.eval()

    epochs_trained, steps_in_epoch_trained = load_checkpoint(model, logger, args)

    optimizer = torch.optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.learning_rate * session.get_world_size(),
    )
    optimizer = train.torch.prepare_optimizer(optimizer)

    address = "localhost:9999"
    s = Server(address, args.data_dir, 0, len(args.partitions))
    dataset = config["init_dataset_fn"](
        args,
        model,
        rank=session.get_world_rank(),
        world_size=session.get_world_size(),
    )
    losses_full = []
    epoch_iter = (
        range(args.num_epochs)
        if not hasattr(dataset, "iter_epochs")
        else dataset.iter_epochs()
    )
    for epoch, epoch_pipe in enumerate(epoch_iter):
        scores = []
        labels = []
        losses = []
        batch_iter = (
            dataset
            if isinstance(epoch_pipe, int)
            else epoch_pipe.iter_torch_batches(batch_size=args.batch_size)
        )
        for step, batch in enumerate(batch_iter):
            if step < steps_in_epoch_trained:
                continue
            loss, score, label = model(batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            scores.append(score)
            labels.append(label)
            losses.append(loss.item())

        losses_full.extend(losses)
        steps_in_epoch_trained = 0
        if epoch % args.save_ckpt_by_epochs == 0:
            save_checkpoint(model, logger, epoch, step, args)

        session.report(
            {
                "metric": model.compute_metric(scores, labels).item(),
                "loss": np.mean(losses),
                "losses": losses_full,
            },
        )


def run_ray(init_dataset_fn, **kwargs):
    """Run ray trainer."""
    ray.init(num_cpus=4)

    args = get_args(init_args, kwargs["run_args"] if "run_args" in kwargs else None)

    trainer = TorchTrainer(
        train_func,
        train_loop_config={
            "args": args,
            "init_dataset_fn": init_dataset_fn,
            **kwargs,
        },
        scaling_config=ScalingConfig(
            num_workers=1, use_gpu=args.gpu, resources_per_worker={"CPU": 2}
        ),
    )
    return trainer.fit()


def _main():
    # setup default logging component.
    setup_default_logging_config(enable_telemetry=True)

    # run_dist is the unified entry for pytorch model distributed training/evaluation/inference.
    # User only needs to prepare initializing function for model, dataset, optimizer and args.
    # reference: `deepgnn/pytorch/training/factory.py`
    run_ray(
        init_dataset_fn=create_dataset,
    )


if __name__ == "__main__":
    _main()
