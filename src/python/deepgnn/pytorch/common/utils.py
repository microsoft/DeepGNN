# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Common function used accross different models."""

import numpy as np
import os
import random
import re
from typing import Tuple, List
import torch

from pathlib import Path
from urllib.parse import urlparse

from deepgnn import get_logger
from deepgnn.pytorch.common.consts import PREFIX_CHECKPOINT
from deepgnn.graph_engine import FeatureType


def get_feature_type(feature_type_str: str) -> FeatureType:
    """Convert string to feature type Enum."""
    feature_type_str = feature_type_str.lower()
    if feature_type_str == "float":
        return FeatureType.FLOAT
    if feature_type_str == "uint64":
        return FeatureType.INT64
    if feature_type_str == "binary":
        return FeatureType.BINARY
    raise RuntimeError(f"Unknown feature type:{feature_type_str}")


def set_seed(seed: int):
    """
    Set the random seed in ``random``, ``numpy`` and ``torch`` modules.

    Args:
        seed: The seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_cuda(context):
    """
    Move all tensor data in context to cuda.

    Args:
        context: nested tensor/numpy array dictionary.
    """
    if isinstance(context, dict):
        for key in context:
            data = context[key]
            if (
                isinstance(data, dict)
                or isinstance(data, tuple)
                or isinstance(data, list)
            ):
                to_cuda(data)
            elif isinstance(data, torch.Tensor):
                context[key] = data.cuda()
            else:
                raise RuntimeError(
                    f"Failed to move {data} to cuda as the type is not unsupported."
                )
    elif isinstance(context, tuple) or isinstance(context, list):
        for i in range(len(context)):
            data = context[i]
            if (
                isinstance(data, dict)
                or isinstance(data, tuple)
                or isinstance(data, list)
            ):
                to_cuda(data)
            # for types which can be converted to cuda, call data.cuda() to convert it,
            # for others, keep it as original.
            elif isinstance(data, torch.Tensor):
                context[i] = data.cuda()


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
