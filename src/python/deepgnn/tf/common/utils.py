# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Common utility functions for training."""
import os
import subprocess
import time
import glob
from typing import List, Tuple, Optional
import numpy as np
import tensorflow as tf

from deepgnn.tf.common.hooks import SessionExitHook, ChiefCheckpointSaverHook
from deepgnn import get_logger


def reset_tf_graph():
    """Clear global default graph."""
    # intentionally move import here.
    # legacy code: deepgnn.tf.layers.
    from deepgnn.tf.layers import base
    import collections

    base._LAYER_UIDS = collections.defaultdict(lambda: 0)
    tf.compat.v1.reset_default_graph()


def setup_worker_hooks(
    model_dir: str,
    task_index: int,
    logging_tensor: dict = {},
    summary_tensor: dict = {},
    dist_sync: ChiefCheckpointSaverHook = None,
    metric_dir: str = "train",
    log_save_steps: int = 20,
    summary_save_steps: int = 100,
    profiler_save_secs: int = 180,
) -> List[tf.estimator.SessionRunHook]:
    """
    Create tensorflow session hooks.

    Args:
      * model_dir: output directory for logging tensor and summary saver.
      * task_index: if task_index == 0, add tf.train.ProfilerHook
      * logging_tensor: tf.train.LoggingTensorHook
      * summary_tensor: tf.train.SummarySaverHook
      * dist_sync: deepgnn.tf.common.hooks.SessionExitHook
    """
    hooks = []
    # Logging tensor hook
    if len(logging_tensor) > 0:
        hooks.append(
            tf.estimator.LoggingTensorHook(logging_tensor, every_n_iter=log_save_steps)
        )
    # summary saver hook
    if len(summary_tensor) > 0:
        ops = [
            tf.compat.v1.summary.scalar(name=k, tensor=summary_tensor[k])
            for k in summary_tensor
        ]
        metric_file = os.path.join(
            model_dir, metric_dir, "worker_{0}".format(task_index)
        )
        get_logger().info(f"summary file: {metric_file}")
        writer = tf.compat.v1.summary.FileWriter(metric_file)
        hooks.append(
            tf.estimator.SummarySaverHook(
                save_steps=summary_save_steps, summary_writer=writer, summary_op=ops
            )
        )
    # profile hook
    if task_index == 0:
        profile_dir = os.path.join(model_dir, "profile")
        hooks.append(
            tf.estimator.ProfilerHook(
                save_secs=profiler_save_secs, output_dir=profile_dir
            )
        )

    if dist_sync is not None:
        hooks.append(SessionExitHook(dist_sync))

    return hooks


def setup_chief_only_hooks(
    task_index: int,
    checkpoint_dir: str,
    dist_sync: ChiefCheckpointSaverHook = None,
    checkpoint_save_secs: int = 3600,
) -> Optional[List[ChiefCheckpointSaverHook]]:
    """
    Create chief only hooks.

    Args:
      * task_index: if task_index == 0, return None
      * checkpoint_dir: save model checkpoint to this directory.
      * dist_sync: deepgnn.tf.common.ChiefCheckpointSaverHook
    """
    if task_index != 0 or dist_sync is None:
        return None

    chief_only_hooks = [
        ChiefCheckpointSaverHook(
            dist_sync, checkpoint_dir=checkpoint_dir, save_secs=checkpoint_save_secs
        )
    ]
    return chief_only_hooks


def node_embedding_to_string(
    embedding_tensor_list: Tuple[tf.Tensor, tf.Tensor], invalid_id: int = -1
):
    """
    Convert node embedding to output string.

    Args:
      - embedding_tensor_list: a list of two tensors.
        * 1st tensor is node id, tensor shape: (batch_size, )
        * 2nd tensor is embedding, tensor shape: (batch_size, dim)
      - invalid_id: skip this node embedding if node id is invalid (default: -1).
    """
    assert len(embedding_tensor_list) == 2
    nodes, embedding = embedding_tensor_list
    assert nodes.ndim == 1 and embedding.ndim == 2
    assert embedding.shape[0] == nodes.shape[0]
    res = []
    idx = np.where(nodes != invalid_id)
    for nodei, embi in zip(nodes[idx], embedding[idx]):
        res.append(
            "{0}\t{1}\n".format(nodei, " ".join(["{:.6f}".format(ei) for ei in embi]))
        )
    return res


def run_commands(commands: List[str]):
    """Run commands in a separate process."""
    get_logger().info(commands)
    proc = subprocess.Popen(args=commands, shell=True)
    while proc.poll() is None:
        time.sleep(3)
    return proc.poll()


def load_embeddings(
    model_dir: str, num_nodes: int, dim: int, fileprefix: str = "embedding_*.tsv"
):
    """Load embeddings from files identified by prefix."""
    res = np.zeros((num_nodes, dim), dtype=np.float32)
    files = glob.glob(os.path.join(model_dir, fileprefix))
    for fname in files:
        for line in open(fname):
            col = line.split("\t")
            nid = int(col[0])
            emb = [float(c) for c in col[1].split(" ")]
            assert len(emb) == dim
            res[nid] = emb
    return res


def get_metrics_from_event_file(model_dir: str, metric_dir: str, metric_name: str):
    """Extract a list of metric values from event file."""
    events_file_pattern = os.path.join(model_dir, metric_dir, "events*")
    events_files = sorted(glob.glob(events_file_pattern))
    get_logger().info(f"event files: {events_files}")
    metric_values = []
    for evt in tf.compat.v1.train.summary_iterator(events_files[0]):
        for v in evt.summary.value:
            if v.tag == metric_name:
                metric_values.append(v.simple_value)
    return metric_values


def log_model_info(model: tf.keras.Model, use_tf_compat: bool = True):
    """Print model's internal variables."""
    # fmt: off
    logger = get_logger()

    def _tf2_log():
        logger.info("--------------variables-----------")
        for i, t in enumerate(model.variables):
            logger.info(f"{i}\t{t.name}\t{t.shape}")
        logger.info("--------------trainable variables-----------")
        for i, t in enumerate(model.trainable_variables):
            logger.info(f"{i}\t{t.name}\t{t.shape}")
        logger.info("--------------------------------------------")

    def _tf1_log():
        logger.info("-------------Local variables-----------------------")
        for v in tf.compat.v1.local_variables():
            logger.info("\t".join(["", str(v.device), str(v.dtype), str(v.get_shape()), v.name]))
        logger.info("-------------Global variables----------------------")
        for v in tf.compat.v1.global_variables():
            logger.info("\t".join(["", str(v.device), str(v.dtype), str(v.get_shape()), v.name]))
        logger.info("-------------Trainable Variables-------------------")
        for v in tf.compat.v1.trainable_variables():
            logger.info("\t".join(["", str(v.device), str(v.dtype), str(v.get_shape()), v.name]))

    if use_tf_compat:
        _tf1_log()
    else:
        _tf2_log()
