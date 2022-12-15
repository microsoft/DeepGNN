"""Baseline Ray Trainer."""
from typing import Dict
import platform
import numpy as np
import torch
import ray
import ray.train as train
from ray.train.torch import TorchTrainer
from ray.air import session
from ray.air.config import ScalingConfig
from deepgnn import TrainMode
from deepgnn.graph_engine import create_backend, BackendOptions
from deepgnn.graph_engine.samplers import GENodeSampler, GEEdgeSampler
from deepgnn.pytorch.common import get_args


def train_func(config: Dict):
    """Training loop for ray trainer."""
    train.torch.enable_reproducibility(seed=session.get_world_rank())

    args = config["args"]

    if args.fp16:
        train.accelerate(True)

    model = config["init_model_fn"](args)
    model = train.torch.prepare_model(model, move_to_device=args.gpu)
    if args.mode == TrainMode.TRAIN:
        model.train()
    else:
        model.eval()

    optimizer = config["init_optimizer_fn"](
        args,
        model,
        session.get_world_size(),
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

            if i >= len(dataset) / args.batch_size / session.get_world_size():
                break

        session.report(
            {
                "metric": model.compute_metric(scores, labels).item(),
                "loss": np.mean(losses),
            }
        )


def run_ray(init_model_fn, init_dataset_fn, init_optimizer_fn, init_args_fn, **kwargs):
    """Run ray trainer."""
    ray.init()

    args = get_args(
        init_args_fn, kwargs["run_args"] if "run_args" in kwargs else None
    )

    trainer = TorchTrainer(
        train_func,
        train_loop_config={
            "args": args,
            "init_model_fn": init_model_fn,
            "init_dataset_fn": init_dataset_fn,
            "init_optimizer_fn": init_optimizer_fn,
            **kwargs,
        },
        scaling_config=ScalingConfig(
            num_workers=1, use_gpu=args.gpu, resources_per_worker={"CPU": 2}
        ),
    )
    trainer.fit()
