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
from deepgnn.pytorch.common.ray_train import run_ray
from deepgnn.graph_engine import TextFileSampler, GraphEngineBackend
from args import init_args  # type: ignore
from consts import DEFAULT_VOCAB_CHAR_INDEX  # type: ignore
from model import LinkPredictionModel  # type: ignore


def create_model(args: argparse.Namespace):
    get_logger().info(f"Creating LinkPredictionModel with seed:{args.seed}.")
    # set seed before instantiating the model

    feature_enc = get_feature_encoder(args)

    return LinkPredictionModel(
        args=args,
        feature_dim=args.feature_dim,
        feature_idx=args.feature_idx,
        feature_type=get_python_type(args.feature_type),
        feature_enc=feature_enc[0],  # type: ignore
        vocab_index=feature_enc[1][DEFAULT_VOCAB_CHAR_INDEX],  # type: ignore
    )


def create_dataset(
    args: argparse.Namespace,
    model: BaseModel,
    rank: int = 0,
    world_size: int = 1,
    address: str = None,
):
    g = DistributedClient([address])
    max_id = g.node_count(args.node_type) if args.max_id in [-1, None] else args.max_id
    dataset = ray.data.range(max_id).repartition(max_id // args.batch_size)
    pipe = dataset.window(blocks_per_window=4).repeat(args.num_epochs)

    def transform_batch(idx: list) -> dict:
        return model.q.query_training(g, np.array(idx))

    pipe = pipe.map_batches(transform_batch)
    return pipe



def create_optimizer(args: argparse.Namespace, model: BaseModel, world_size: int):
    return torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate * world_size,
        weight_decay=args.weight_decay,
    )


def _main():
    # setup default logging component.
    setup_default_logging_config(enable_telemetry=True)

    # run_dist is the unified entry for pytorch model distributed training/evaluation/inference.
    # User only needs to prepare initializing function for model, dataset, optimizer and args.
    # reference: `deepgnn/pytorch/training/factory.py`
    run_ray(
        init_model_fn=create_model,
        init_dataset_fn=create_dataset,
        init_optimizer_fn=create_optimizer,
        init_args_fn=init_args,
    )


if __name__ == "__main__":
    _main()
