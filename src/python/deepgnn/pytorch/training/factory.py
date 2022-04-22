# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
import torch
from typing import Optional, Callable, List
from deepgnn import TrainerType
from deepgnn import get_logger
from contextlib import closing
from deepgnn.pytorch.common import init_common_args
from deepgnn.pytorch.training.args import init_trainer_args, init_fp16_args
from deepgnn.pytorch.training.trainer import Trainer
from deepgnn.graph_engine import create_backend, BackendOptions
from deepgnn.graph_engine.samplers import GENodeSampler, GEEdgeSampler


def get_args(init_arg_fn: Optional[Callable] = None, run_args: Optional[List] = None):
    parser = argparse.ArgumentParser(allow_abbrev=False)

    # Initialize common parameters, including model, dataset, optimizer etc.
    init_common_args(parser)

    # Initialize trainer paramaters.
    init_trainer_args(parser)

    # Initialize fp16 related paramaters.
    init_fp16_args(parser)

    if init_arg_fn is not None:
        init_arg_fn(parser)

    args = parser.parse_args() if run_args is None else parser.parse_args(run_args)
    for arg in dir(args):
        if not arg.startswith("_"):
            get_logger().info(f"{arg}={getattr(args, arg)}")

    return args


def get_trainer(args: argparse.Namespace) -> Trainer:
    if args.trainer == TrainerType.BASE:
        return Trainer(args)

    elif args.trainer == TrainerType.DDP:
        from deepgnn.pytorch.training.trainer_ddp import DDPTrainer

        return DDPTrainer(args)
    elif args.trainer == TrainerType.HVD:
        from deepgnn.pytorch.training.trainer_hvd import HVDTrainer

        return HVDTrainer(args)
    else:
        raise RuntimeError(f"Unknown trainer type: {args.trainer}.")


def run_dist(
    init_model_fn: Callable,
    init_dataset_fn: Callable,
    init_optimizer_fn: Optional[Callable] = None,
    init_args_fn: Optional[Callable] = None,
    run_args: Optional[List] = None,
    init_eval_dataset_for_training_fn: Optional[Callable] = None,
):
    """Run distributed training/evaluation/inference.

    Args:
    init_model_fn: (`Callable[args:argparse.Namespace]`)
        Function to initialize gnn model.
    init_dataset_fn: (`Callable[args:argparse.Namespace, graph:Graph, model:BaseModel, rank:int, world_size:int]`)
        Function to initialize dataset.
    init_optimizer_fn: (`Callable[args:argparse.Namespace, model:BaseModel, world_size:int]`, `optional`)
        Function to initialize optimizer, not needed for evaluation/inference.
    init_args_fn: (`Callable[args:argparse.ArgumentParser]`, `optional`)
        Function to add or override command line arguments.
    run_args: (`List[str]`, `optional`)
        List of arguments to pass to argument parser in place of sys.argv, in format ['--data_dir', 'path/to', ...].
    init_eval_dataset_for_training_fn: (`Callable[args:argparse.Namespace, graph:Graph, model:BaseModel, rank:int, world_size:int]`, `optional`)
        Function to initialize evaluation dataset during training.
    """
    args = get_args(init_args_fn, run_args)
    trainer = get_trainer(args)
    backend = create_backend(BackendOptions(args), is_leader=(trainer.rank == 0))

    model = init_model_fn(args)
    dataset = init_dataset_fn(
        args=args,
        model=model,
        rank=trainer.rank,
        world_size=trainer.world_size,
        backend=backend,
    )

    eval_dataloader_for_training = None
    if init_eval_dataset_for_training_fn is not None:
        eval_dataset_for_training = init_eval_dataset_for_training_fn(
            args=args,
            model=model,
            rank=trainer.rank,
            world_size=trainer.world_size,
            backend=backend,
        )
        if eval_dataset_for_training is not None:
            eval_dataloader_for_training = torch.utils.data.DataLoader(
                dataset=eval_dataset_for_training,
                num_workers=args.data_parallel_num,
                prefetch_factor=args.prefetch_factor,
            )
    optimizer = (
        init_optimizer_fn(args=args, model=model, world_size=trainer.world_size)
        if init_optimizer_fn is not None
        else None
    )

    num_workers = (
        0
        if issubclass(dataset.sampler_class, (GENodeSampler, GEEdgeSampler))
        else args.data_parallel_num
    )

    # Executed distributed training/evalution/inference.
    with closing(backend):
        trainer.run(
            model,
            torch.utils.data.DataLoader(
                dataset=dataset,
                num_workers=num_workers,
                prefetch_factor=args.prefetch_factor,
            ),
            optimizer,
            eval_dataloader_for_training,
        )
