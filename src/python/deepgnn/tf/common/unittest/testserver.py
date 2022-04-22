# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
import tensorflow as tf
from deepgnn.tf.common import DistributedSync, SessionExitHook, ChiefCheckpointSaverHook
from deepgnn.tf import layers
import numpy as np

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
tf.compat.v1.disable_v2_behavior()


def str2list(v):
    return v.split(",")


def define_param_ps_dist_training(parser):
    ## ps/worker_host
    parser.add_argument(
        "--job_name",
        type=str,
        default="worker",
        choices=["worker", "ps"],
        help="Job role.",
    )
    parser.add_argument("--task_index", type=int, default=0, help="Job index.")
    parser.add_argument(
        "--ps_hosts", type=str2list, default="", help="parameter servers."
    )
    parser.add_argument(
        "--worker_hosts", type=str2list, default="", help="training/inference workers."
    )
    parser.add_argument("--model_dir", type=str, default="/tmp/test123")


def get_dist_training_server(param):
    assert len(param.ps_hosts) > 0
    assert len(param.worker_hosts) > 0
    clustersepc = tf.train.ClusterSpec(
        {"ps": param.ps_hosts, "worker": param.worker_hosts}
    )
    server = tf.distribute.Server(
        clustersepc, job_name=param.job_name, task_index=param.task_index
    )
    return server, clustersepc


def log_all_parameters(args, logger):
    for arg in vars(args):
        logger("{0}:\t{1}".format(arg, getattr(args, arg)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter, allow_abbrev=False
    )
    define_param_ps_dist_training(parser)
    param = parser.parse_args()
    log_all_parameters(param, tf.compat.v1.logging.info)

    model_dir = param.model_dir

    tfserver, cluster = get_dist_training_server(param)
    if param.job_name == "ps":
        tfserver.join()

    with tf.compat.v1.device(
        tf.compat.v1.train.replica_device_setter(
            cluster=cluster,
            worker_device="/job:worker/task:{}".format(param.task_index),
        )
    ):
        v = np.arange(0, 100, dtype=np.float32)
        v = np.reshape(v, (-1, 2))
        ds = tf.data.Dataset.from_tensors(v)
        ds = ds.repeat(10)
        x = tf.compat.v1.data.make_one_shot_iterator(ds).get_next()
        w = layers.Dense(1, use_bias=False)
        y = w(x)
        loss = tf.reduce_sum(input_tensor=y)
        global_step = tf.compat.v1.train.get_or_create_global_step()
        optimizer = tf.compat.v1.train.AdamOptimizer(0.001)
        train_op = optimizer.minimize(loss, global_step=global_step)
        dist_sync = DistributedSync(
            model_dir, param.task_index, len(param.worker_hosts)
        )
        hooks = []
        hooks.append(SessionExitHook(dist_sync))
        hooks.append(
            tf.estimator.LoggingTensorHook({"step": global_step}, every_n_iter=1)
        )
        chiefhooks = []
        if param.task_index == 0:
            chiefhooks.append(
                ChiefCheckpointSaverHook(dist_sync, model_dir, save_secs=1)
            )

        with tf.compat.v1.train.MonitoredTrainingSession(
            master=tfserver.target,
            is_chief=True if param.task_index == 0 else False,
            save_checkpoint_secs=None,
            save_checkpoint_steps=None,
            log_step_count_steps=None,
            hooks=hooks,
            chief_only_hooks=chiefhooks,
        ) as sess:
            while not sess.should_stop():
                op = sess.run([train_op])
        dist_sync.sync("final")
