# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
import os

from deepgnn import TrainMode, setup_default_logging_config
from deepgnn.pytorch.common import create_adamw_optimizer
from deepgnn.pytorch.common.dataset import TorchDeepGNNDataset
from deepgnn.pytorch.common.utils import (
    get_logger,
    get_python_type,
    get_store_name_and_path,
    set_seed,
)
from deepgnn.pytorch.encoding import get_feature_encoder
from deepgnn.pytorch.modeling import BaseModel
from deepgnn.pytorch.training import run_dist
from deepgnn.graph_engine import TextFileSampler, GraphEngineBackend
from args import init_args
from consts import DEFAULT_VOCAB_CHAR_INDEX
from model import LinkPredictionModel


def create_model(args: argparse.Namespace):
    get_logger().info(f"Creating LinkPredictionModel with seed:{args.seed}.")
    # set seed before instantiating the model
    if args.seed:
        set_seed(args.seed)

    feature_enc = get_feature_encoder(args)

    return LinkPredictionModel(
        args=args,
        feature_dim=args.feature_dim,
        feature_idx=args.feature_idx,
        feature_type=get_python_type(args.feature_type),
        feature_enc=feature_enc[0],
        vocab_index=feature_enc[1][DEFAULT_VOCAB_CHAR_INDEX],
    )


def create_dataset(
    args: argparse.Namespace,
    model: BaseModel,
    rank: int = 0,
    world_size: int = 1,
    backend: GraphEngineBackend = None,
):
    store_name, relative_path = get_store_name_and_path(args.train_file_dir)

    return TorchDeepGNNDataset(
        sampler_class=TextFileSampler,
        backend=backend,
        query_fn=model.query,
        prefetch_queue_size=10,
        prefetch_worker_size=2,
        batch_size=args.batch_size,
        store_name=store_name,
        filename=os.path.join(relative_path, "*"),
        adl_config=args.adl_config,
        shuffle=False,
        drop_last=False,
        worker_index=rank,
        num_workers=world_size,
        epochs=-1 if (args.max_samples > 0 and args.mode == TrainMode.TRAIN) else 1,
        buffer_size=1024,
    )


def create_optimizer(args: argparse.Namespace, model: BaseModel, world_size: int):
    return create_adamw_optimizer(
        model, args.weight_decay, args.learning_rate * world_size
    )


def _main():
    # setup default logging component.
    setup_default_logging_config(enable_telemetry=True)

    # run_dist is the unified entry for pytorch model distributed training/evaluation/inference.
    # User only needs to prepare initializing function for model, dataset, optimizer and args.
    # reference: `deepgnn/pytorch/training/factory.py`
    run_dist(
        init_model_fn=create_model,
        init_dataset_fn=create_dataset,
        init_optimizer_fn=create_optimizer,
        init_args_fn=init_args,
    )


if __name__ == "__main__":
    _main()
