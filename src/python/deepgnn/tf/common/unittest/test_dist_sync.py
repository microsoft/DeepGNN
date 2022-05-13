# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import subprocess
import tensorflow as tf
import tempfile
import time
from deepgnn.tf.common.utils import reset_tf_graph


def check_process_status(processes):
    error = False
    finished = False
    stat = [p.poll() for p in processes]
    error_processes = []

    details = []
    for i in range(0, len(processes)):
        s = "."
        if stat[i] is not None:
            s = stat[i]
        if stat[i] is not None and stat[i] != 0:
            error = True
            error_processes.append(processes[i])
        details.append("({0} {1})".format(processes[i].pid, s))

    if all([s is not None for s in stat]):
        finished = True
    return error, finished, "|".join(details), error_processes


def test_dist_training():
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
    model_path = tempfile.TemporaryDirectory()
    tf.compat.v1.logging.info("model_dir {}".format(model_path.name))

    ps_count, wk_count = 1, 2
    ps_hosts = ",".join(["localhost:{}".format(1999 - i) for i in range(ps_count)])
    wk_hosts = ",".join(["localhost:{}".format(2000 + i) for i in range(wk_count)])

    proc_ps = []
    current_dir = os.path.dirname(os.path.realpath(__file__))
    testfile = os.path.join(current_dir, "testserver.py")

    for i in range(ps_count):
        cmd = "env CUDA_VISIBLE_DEVICES= python3 {} --job_name=ps --task_index {} --ps_hosts {} --worker_hosts={} --model_dir {}".format(
            testfile, i, ps_hosts, wk_hosts, model_path.name
        )
        tf.compat.v1.logging.info(cmd)
        cmd_args = cmd.split(" ")
        proc_ps.append(subprocess.Popen(args=cmd_args))
    proc_wk = []
    for i in range(wk_count):
        cmd = "env CUDA_VISIBLE_DEVICES= python3 {} --job_name=worker --task_index={} --ps_hosts {} --worker_hosts={} --model_dir {}".format(
            testfile, i, ps_hosts, wk_hosts, model_path.name
        )
        tf.compat.v1.logging.info(cmd)
        cmd_args = cmd.split(" ")
        proc_wk.append(subprocess.Popen(args=cmd_args))

    exit_code = 0
    while True:
        err_code_ps, finished_ps, log_ps, err_process_ps = check_process_status(proc_ps)
        err_code_wk, finished_wk, log_wk, err_process_wk = check_process_status(proc_wk)
        tf.compat.v1.logging.info("ps {0}, workers {1}".format(log_ps, log_wk))
        if err_code_ps or err_code_wk:
            tf.compat.v1.logging.error("some processes exit with non-zero code...")
            exit_code = 1
            break
        if finished_wk:
            tf.compat.v1.logging.info("all workers are finished.")
            break
        time.sleep(1)

    tf.compat.v1.logging.info("terminate all processes...")
    [p.kill() for p in proc_ps if p.poll() is None]
    [p.kill() for p in proc_wk if p.poll() is None]
    assert exit_code == 0

    reset_tf_graph()
    tf.compat.v1.disable_v2_behavior()
    # restore checkpoint and check global step.
    global_step = tf.compat.v1.train.get_or_create_global_step()
    with tf.compat.v1.train.MonitoredTrainingSession(
        checkpoint_dir=model_path.name,
        save_checkpoint_secs=None,
        save_checkpoint_steps=None,
    ) as sess:
        gstep = sess.run(global_step)
        assert gstep == 20

    model_path.cleanup()
