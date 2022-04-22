# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from deepgnn import get_logger
import tensorflow as tf
from deepgnn.graph_engine.backends.common import GraphEngineBackend
from deepgnn.graph_engine.graph_dataset import DeepGNNDataset
from typing import Callable


def create_tf_dataset(
    sampler_class,
    query_fn: Callable = None,
    backend: GraphEngineBackend = None,
    num_workers: int = 1,
    worker_index: int = 0,
    batch_size: int = 1,
    enable_prefetch: bool = False,
    # parameters to initialize samplers
    **kwargs,
):
    def _get_tf_dtypes(tensor_list):
        tf_types = []
        for x in tensor_list:
            x2 = tf.convert_to_tensor(x)
            tf_types.append(x2.dtype)
            get_logger().info(
                f"numpy ({x.shape}, {x.dtype}), tf tensor ({x2.shape}, {x2.dtype})"
            )
        return tf_types

    def _check_array_shapes(nparray_list, expected_shape_list):
        matched = True
        for arr, exp in zip(nparray_list, expected_shape_list):
            shape = arr.shape
            get_logger().info(f"check array shape: {shape} vs {exp}")
            assert len(shape) == len(exp)
            for i in range(len(exp)):
                if exp[i] is not None and shape[i] != exp[i]:
                    matched = False
        return matched

    dataset = DeepGNNDataset(
        sampler_class,
        query_fn,
        backend,
        num_workers,
        worker_index,
        batch_size,
        1,  # set epochs 1 here
        enable_prefetch,
        None,
        **kwargs,
    )

    inputs = next(iter(dataset.sampler))  # type: ignore
    graph_tensor, shapes = dataset.query_fn(  # type: ignore
        dataset.graph, inputs, return_shape=True
    )

    assert len(graph_tensor) == len(shapes)
    assert _check_array_shapes(graph_tensor, shapes)
    tf_shapes = [tf.TensorShape(x) for x in shapes]
    tf_types = _get_tf_dtypes(graph_tensor)

    graph_dataset = tf.data.Dataset.from_generator(
        lambda: iter(dataset), tuple(tf_types), output_shapes=tuple(tf_shapes)
    )

    # NOTE: if dataset is distributed, we need to provide steps_per_epoch for keras API
    # otherwise we can use None.
    try:
        length = len(dataset)
    except:
        # some of the samplers are unable to provide __len__ function
        length = None  # type: ignore

    return (
        graph_dataset if dataset.enable_prefetch else graph_dataset.prefetch(-1),
        length,
    )


def get_distributed_dataset(func: Callable):
    """Distribute_datasets_from_function is introduced in tensorflow 2.4, for
    other tensorflow versions >= 2 and < 2.4, please use experimental_distribute_datasets_from_function
    instead."""
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
