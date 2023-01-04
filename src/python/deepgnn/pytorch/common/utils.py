# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Common function used accross different models."""

import numpy as np
import os
import re
from typing import Tuple, List
import torch

from pathlib import Path
from urllib.parse import urlparse

from deepgnn import get_logger
from deepgnn.pytorch.common.consts import PREFIX_CHECKPOINT


def get_python_type(dtype_str: str):
    """Convert string to feature type Enum."""
    dtype_str = dtype_str.lower()
    if dtype_str == "float":
        return np.float32
    if dtype_str == "uint64":
        return np.int64
    if dtype_str == "binary":
        return np.uint8
    raise RuntimeError(f"Unknown feature type:{dtype_str}")


def get_store_name_and_path(input_path: str) -> Tuple[str, str]:
    """
    Get store name and relative path if input path is adl path, or return input_path directly.

    Args:
        input_path: adl or local directory path.

    Returns:
        store name and relative path.
    """
    if input_path is None or len(input_path) == 0:
        return "", ""

    if input_path.lower().startswith("adl:"):
        o = urlparse(input_path)
        assert o.hostname is not None
        return o.hostname.split(".")[0], o.path

    return "", input_path


def print_model(model: torch.nn.Module):
    """Print model state."""
    state_dict = model.state_dict()
    for i, key in enumerate(state_dict):
        get_logger().info(
            f"{i}, {key}: {state_dict[key].shape}, {state_dict[key].device}"
        )


def tally_parameters(model: torch.nn.Module):
    """Print model named parameters."""
    n_params = 0
    for name, param in model.named_parameters():
        n_params += param.nelement()
    get_logger().info(f"parameter count: {n_params}")


def dump_gpu_memory(prefix="") -> str:
    """Return GPU memory usage statistics."""
    MB = 1024 * 1024
    return (
        f"{prefix} Memory Allocated: {torch.cuda.memory_allocated() / MB:.3f} MB"
        + f" | Max Memory Allocated: {torch.cuda.max_memory_allocated() / MB:.3f} MB"
        + f" | Memory Reserved: {torch.cuda.memory_reserved() / MB:.3f} MB"
        + f" | Max Memory Reserved: {torch.cuda.max_memory_reserved() / MB:.3f} MB"
    )


def get_sorted_checkpoints(
    model_dir: str, perfix: str = PREFIX_CHECKPOINT, sort_ckpt_by_mtime: bool = False
) -> List[str]:
    """Return model checkpoints."""
    ordering_and_checkpoint_path = []

    glob_checkpoints = [str(x) for x in Path(model_dir).glob(f"{perfix}-*")]

    for path in glob_checkpoints:
        if sort_ckpt_by_mtime:
            ordering_and_checkpoint_path.append((str(os.path.getmtime(path)), path))
        else:
            regex_match = re.match(f".*{perfix}-([0-9\\-]+)", path)
            if regex_match and regex_match.groups():
                ordering_and_checkpoint_path.append((regex_match.groups()[0], path))

    return [checkpoint[1] for checkpoint in sorted(ordering_and_checkpoint_path)]


def rotate_checkpoints(
    model_dir: str,
    max_saved_ckpts: int = 0,
    prefix: str = PREFIX_CHECKPOINT,
    sort_ckpt_by_mtime: bool = False,
):
    """Remove old checkpoints if total number of checkpoints has exceeded max_saved_ckpts."""
    if max_saved_ckpts > 0:
        # Check if we should delete old checkpoint(s)
        checkpoints_sorted = get_sorted_checkpoints(
            model_dir, prefix, sort_ckpt_by_mtime
        )

        if len(checkpoints_sorted) > max_saved_ckpts:
            number_of_checkpoints_to_delete = max(
                0, len(checkpoints_sorted) - max_saved_ckpts
            )
            checkpoints_to_be_deleted = checkpoints_sorted[
                :number_of_checkpoints_to_delete
            ]
            for checkpoint in checkpoints_to_be_deleted:
                os.remove(checkpoint)
                get_logger().info(
                    f"Deleted checkpoint [{checkpoint}] due to max_saved_ckpts:{max_saved_ckpts}"
                )


def load_checkpoint(model, logger, args):
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
    return epochs_trained, steps_in_epoch_trained


def save_checkpoint(model, logger, epoch, step, args, **kwargs):
    os.makedirs(args.save_path, exist_ok=True)
    save_path = os.path.join(
        f"{args.save_path}",
        f"{PREFIX_CHECKPOINT}-{epoch:03}-{step:06}.pt",
    )
    torch.save(
        {
            "state_dict": model.state_dict(),
            "epoch": epoch,
            "step": step,
            **kwargs
        },
        save_path,
    )
    rotate_checkpoints(args.save_path, args.max_saved_ckpts)
    logger.info(f"Saved checkpoint to {save_path}.")
