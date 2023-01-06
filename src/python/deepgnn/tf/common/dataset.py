# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Commmon functions to create datasets in TF."""
from typing import Tuple, Optional
from deepgnn import get_logger
import tensorflow as tf
from typing import Callable


def get_distributed_dataset(func: Callable) -> tf.data.Dataset:
    """Generate distributed dataset from function.

    Distribute_datasets_from_function is introduced in tensorflow 2.4. For other tensorflow versions >= 2 and < 2.4,
    please use experimental_distribute_datasets_from_function instead.
    """
    strategy = tf.distribute.get_strategy()
    if hasattr(strategy, "distribute_datasets_from_function"):
        distributed_dataset = strategy.distribute_datasets_from_function(func)
    elif hasattr(strategy, "experimental_distribute_datasets_from_function"):
        distributed_dataset = strategy.experimental_distribute_datasets_from_function(
            func
        )
    else:
        raise ValueError("No valid distribute_datasets_from_function can be found")

    return distributed_dataset
