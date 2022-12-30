"""Baseline Ray Trainer."""
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
from deepgnn.pytorch.common.consts import PREFIX_CHECKPOINT
from deepgnn.pytorch.common.utils import rotate_checkpoints, get_sorted_checkpoints


def train_func(config: Dict):
    """Training loop for ray trainer."""
    args = config["args"]

    logger = get_logger()
    os.makedirs(args.save_path, exist_ok=True)

    train.torch.accelerate(args.fp16)
    if args.seed:
        train.torch.enable_reproducibility(seed=args.seed + session.get_world_rank())

    model = config["init_model_fn"](args)
    # https://docs.ray.io/en/latest/tune/api_docs/trainable.html#function-api-checkpointing
    model = train.torch.prepare_model(model, move_to_device=args.gpu)
    if args.mode == TrainMode.TRAIN:
        model.train()
    else:
        model.eval()

    epochs_trained = 0
    steps_in_epoch_trained = 0
    # Search and sort checkpoints from model path.
    ckpts = get_sorted_checkpoints(args.model_dir)
    ckpt_path = ckpts[-1] if len(ckpts) > 0 else None
    if ckpt_path is not None:
        init_ckpt = torch.load(ckpt_path, map_location="cpu")
        if args.mode == TrainMode.TRAIN:
            epochs_trained = init_ckpt["epoch"]
            steps_in_epoch_trained = init_ckpt["step"]

        if session.get_world_rank() == 0:
            model.load_state_dict(init_ckpt["state_dict"])
            logger.info(
                f"Loaded initial checkpoint: {ckpt_path},"
                f" trained epochs: {epochs_trained}, steps: {steps_in_epoch_trained}"
            )
        del init_ckpt

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
            save_path = os.path.join(
                f"{args.save_path}",
                f"{PREFIX_CHECKPOINT}-{epoch:03}-{step:06}.pt",
            )
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "epoch": epoch,
                    "step": step,
                },
                save_path,
            )
            rotate_checkpoints(args.save_path, args.max_saved_ckpts)
            logger.info(f"Saved checkpoint to {save_path}.")

        session.report(
            {
                "metric": model.compute_metric(scores, labels).item(),
                "loss": np.mean(losses),
                "losses": losses_full,
            },
        )


def run_ray(init_model_fn, init_dataset_fn, init_optimizer_fn, init_args_fn, **kwargs):
    """Run ray trainer."""
    ray.init(num_cpus=2)

    args = get_args(init_args_fn, kwargs["run_args"] if "run_args" in kwargs else None)

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
    return trainer.fit()
