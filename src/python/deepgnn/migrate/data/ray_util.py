from typing import Dict
import argparse
import platform
import torch
from typing import Optional, Callable, List
from deepgnn import TrainerType
from deepgnn import get_logger
from contextlib import closing
import numpy as np
import torch.nn as nn
import ray
import ray.train as train
from ray.train.torch import TorchTrainer
from ray.air import session
from ray.air.config import ScalingConfig, RunConfig
from deepgnn.pytorch.common import init_common_args
from deepgnn.pytorch.training.args import init_trainer_args, init_fp16_args
from deepgnn.pytorch.training.trainer import Trainer
from deepgnn.graph_engine import create_backend, BackendOptions
from deepgnn.graph_engine.samplers import GENodeSampler, GEEdgeSampler
from deepgnn.graph_engine.snark.local import Client


# TODO put in util
def get_args(init_arg_fn: Optional[Callable] = None, run_args: Optional[List] = None):
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(allow_abbrev=False)

    # Initialize common parameters, including model, dataset, optimizer etc.
    init_common_args(parser)

    # Initialize trainer paramaters.
    init_trainer_args(parser)

    # Initialize fp16 related paramaters.
    init_fp16_args(parser)

    if init_arg_fn is not None:
        init_arg_fn(parser)

    args, _ = parser.parse_known_args() if run_args is None else parser.parse_known_args(run_args)
    for arg in dir(args):
        if not arg.startswith("_"):
            get_logger().info(f"{arg}={getattr(args, arg)}")

    return args


def train_func(config: Dict):
    train.torch.enable_reproducibility(seed=session.get_world_rank())

    args = config["args"]

    model = config["init_model_fn"](args)
    model = train.torch.prepare_model(model)  # TODO any relevant args
    model.train()

    optimizer = config["init_optimizer_fn"](
        args,
        model,
        session.get_world_size(),
    )
    optimizer = train.torch.prepare_optimizer(optimizer)  # TODO any relevant args

    backend = create_backend(BackendOptions(args), is_leader=(session.get_local_rank() == 0))  # TODO local or world
    dataset = config["init_dataset_fn"](
        args,
        model,
        rank=session.get_local_rank(),  # TODO local or world
        world_size=session.get_world_size(),
        backend=backend,
    )
    num_workers = (
        0
        if issubclass(dataset.sampler_class, (GENodeSampler, GEEdgeSampler))
        or platform.system() == "Windows"
        else args.data_parallel_num
    )
    dataset = torch.utils.data.DataLoader(
        dataset=dataset,
        num_workers=num_workers,
    )
    for epoch in range(args.num_epochs):
        scores = []
        labels = []
        losses = []
        for i, batch in enumerate(dataset):
            loss, score, label = model(batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            scores.append(score)
            labels.append(label)
            losses.append(loss.item())

            #if i >= SAMPLE_NUM / BATCH_SIZE / session.get_world_size():
            #    break

        session.report(
            {
                "metric": model.compute_metric(scores, labels).item(),
                "loss": np.mean(losses),
            }
        )


def run_ray(
    init_model_fn,
    init_dataset_fn,
    init_optimizer_fn,
    init_args_fn,
    **kwargs
):
    ray.init()

    trainer = TorchTrainer(
        train_func,
        train_loop_config={
            "args": get_args(init_args_fn, kwargs["run_args"] if "run_args" in kwargs else None),
            "init_model_fn": init_model_fn,
            "init_dataset_fn": init_dataset_fn,
            "init_optimizer_fn": init_optimizer_fn,
            **kwargs
        },
        run_config=RunConfig(),
        scaling_config=ScalingConfig(num_workers=1, use_gpu=False, resources_per_worker={"CPU": 2}),
    )
    result = trainer.fit()
