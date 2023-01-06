# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
import os
import torch.optim

from deepgnn import TrainMode, setup_default_logging_config
from deepgnn.pytorch.common.utils import (
    get_logger,
    get_python_type,
    get_store_name_and_path,
)
from deepgnn.pytorch.encoding import get_feature_encoder
from deepgnn.pytorch.modeling import BaseModel
from args import init_args  # type: ignore
from consts import DEFAULT_VOCAB_CHAR_INDEX  # type: ignore
from model import LinkPredictionModel  # type: ignore
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


def train_func(config: Dict):
    """Training loop for ray trainer."""
    args = config["args"]

    logger = get_logger()
    os.makedirs(args.save_path, exist_ok=True)

    train.torch.accelerate(args.fp16)
    if args.seed:
        train.torch.enable_reproducibility(seed=args.seed + session.get_world_rank())

    feature_enc = get_feature_encoder(args)
    model = LinkPredictionModel(
        args=args,
        feature_dim=args.feature_dim,
        feature_idx=args.feature_idx,
        feature_type=get_python_type(args.feature_type),
        feature_enc=feature_enc[0],  # type: ignore
        vocab_index=feature_enc[1][DEFAULT_VOCAB_CHAR_INDEX],  # type: ignore
    )
    # https://docs.ray.io/en/latest/tune/api_docs/trainable.html#function-api-checkpointing
    model = train.torch.prepare_model(model, move_to_device=args.gpu)
    if args.mode == TrainMode.TRAIN:
        model.train()
    else:
        model.eval()

    epochs_trained, steps_in_epoch_trained = load_checkpoint(model, logger, args)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate * session.get_world_size(),
        weight_decay=args.weight_decay,
    )
    optimizer = train.torch.prepare_optimizer(optimizer)

    address = "localhost:9999"
    s = Server(address, args.data_dir, 0, len(args.partitions))
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

    for epoch, epoch_pipe in enumerate(pipe.iter_epochs()):
        if epoch < epochs_trained:
            continue
        scores = []
        labels = []
        losses = []
        for step, batch in enumerate(epoch_pipe.iter_torch_batches(batch_size=args.batch_size)):
            loss, score, label = model(batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            scores.append(score)
            labels.append(label)
            losses.append(loss.item())

        steps_in_epoch_trained = 0
        if epoch % args.save_ckpt_by_epochs == 0:
            save_checkpoint(model, logger, epoch, step, args)

        session.report(
            {
                "metric": model.compute_metric(scores, labels).item(),
                "loss": np.mean(losses),
            },
        )


def run_ray(**kwargs):
    """Run ray trainer."""
    ray.init()

    args = get_args(init_args, kwargs["run_args"] if "run_args" in kwargs else None)

    trainer = TorchTrainer(
        train_func,
        train_loop_config={
            "args": args,
            **kwargs,
        },
        scaling_config=ScalingConfig(
            num_workers=1, use_gpu=args.gpu, resources_per_worker={"CPU": 2}
        ),
    )
    trainer.fit()


def _main():
    # setup default logging component.
    setup_default_logging_config(enable_telemetry=True)

    # run_dist is the unified entry for pytorch model distributed training/evaluation/inference.
    # User only needs to prepare initializing function for model, dataset, optimizer and args.
    # reference: `deepgnn/pytorch/training/factory.py`
    run_ray()


if __name__ == "__main__":
    _main()
