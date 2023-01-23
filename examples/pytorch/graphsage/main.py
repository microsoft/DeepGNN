# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
import torch
import numpy as np
from deepgnn import TrainMode, setup_default_logging_config
from deepgnn import get_logger
from deepgnn.pytorch.common import F1Score
from deepgnn.pytorch.common.dataset import TorchDeepGNNDataset
from deepgnn.pytorch.common.utils import get_python_type
from deepgnn.pytorch.encoding import get_feature_encoder
from deepgnn.pytorch.modeling import BaseModel
from deepgnn.graph_engine import (
    Graph,
    SamplingStrategy,
    GENodeSampler,
    GraphEngineBackend,
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
from deepgnn.graph_engine import create_backend, BackendOptions
from deepgnn.graph_engine.samplers import GENodeSampler, GEEdgeSampler
from deepgnn.pytorch.common import get_args
from deepgnn.pytorch.common.utils import load_checkpoint, save_checkpoint
from typing import Optional


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
    backend: Optional[GraphEngineBackend] = None,
):
    if args.mode == TrainMode.INFERENCE:
        return TorchDeepGNNDataset(
            sampler_class=CSVNodeSampler,
            backend=backend,
            num_workers=world_size,
            worker_index=rank,
            batch_size=args.batch_size,
            sample_file=args.sample_file,
            query_fn=model.query,
            prefetch_queue_size=10,
            prefetch_worker_size=2,
        )
    else:
        return TorchDeepGNNDataset(
            sampler_class=GENodeSampler,
            backend=backend,
            sample_num=args.max_id,
            num_workers=world_size,
            worker_index=rank,
            node_types=np.array([args.node_type], dtype=np.int32),
            batch_size=args.batch_size,
            query_fn=model.query,
            prefetch_queue_size=10,
            prefetch_worker_size=2,
            strategy=SamplingStrategy.RandomWithoutReplacement,
        )


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

    epochs_trained, steps_in_epoch_trained = load_checkpoint(
        model, logger, args, session.get_world_rank()
    )

    optimizer = torch.optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.learning_rate * session.get_world_size(),
    )
    optimizer = train.torch.prepare_optimizer(optimizer)

    backend = create_backend(
        BackendOptions(args), is_leader=(session.get_world_rank() == 0)
    )
    dataset = config["init_dataset_fn"](
        args,
        model,
        rank=session.get_world_rank(),
        world_size=session.get_world_size(),
        backend=backend,
    )
    dataset = torch.utils.data.DataLoader(
        dataset=dataset,
        num_workers=0,
    )
    losses_full = []
    for epoch in range(epochs_trained, args.num_epochs):
        scores = []
        labels = []
        losses = []
        for step, batch in enumerate(dataset):
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
    ray.init(num_cpus=3)

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
