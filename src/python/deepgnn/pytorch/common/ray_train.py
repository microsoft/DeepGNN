"""Baseline Ray Trainer."""
from typing import Dict
import os
import numpy as np
import ray
import ray.train as train
from ray.train.torch import TorchTrainer
from ray.air import session
from ray.air.config import ScalingConfig
from deepgnn import TrainMode, get_logger
from deepgnn.pytorch.common import get_args
from deepgnn.pytorch.common.utils import load_checkpoint, save_checkpoint
from deepgnn.graph_engine.snark.distributed import Server


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

    epochs_trained, steps_in_epoch_trained = load_checkpoint(
        model, logger, args, session.get_world_rank()
    )

    optimizer = config["init_optimizer_fn"](
        args,
        model,
        session.get_world_size(),
    )
    optimizer = train.torch.prepare_optimizer(optimizer)

    address = "localhost:9999"
    _ = Server(address, args.data_dir, 0, len(args.partitions))
    pipe = config["init_dataset_fn"](
        args, model, session.get_world_rank(), session.get_world_size(), address
    )

    for epoch, epoch_pipe in enumerate(pipe.iter_epochs()):
        if epoch < epochs_trained:
            continue
        scores = []
        labels = []
        losses = []
        for step, batch in enumerate(
            epoch_pipe.iter_torch_batches(batch_size=args.batch_size)
        ):
            if step < steps_in_epoch_trained:
                continue
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


def run_ray(
    init_model_fn,
    init_dataset_fn,
    init_optimizer_fn,
    init_args_fn,
    num_cpus=None,
    **kwargs
):
    """Run ray trainer."""
    ray.init(num_cpus=num_cpus)

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
