# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import tensorflow as tf
import logging

from deepgnn.tf.common.args import TrainerType
from deepgnn.tf.common.dist_sync import DistributedSync
from deepgnn.tf.common.trainer import BaseTFTrainer


class PSTrainer(BaseTFTrainer):
    def __init__(
        self,
        trainer: TrainerType,
        model_dir: str,
        seed: int,
        ps_hosts: str,
        job_name: str,
        worker_hosts: str,
        task_index: int = 0,
        user_name: str = "",
        job_id: str = "",
        gpu: bool = False,
        log_save_steps: int = 20,
        summary_save_steps: int = 100,
        profiler_save_secs: int = 180,
        checkpoint_save_secs: int = 3600,
        logger: logging.Logger = None,
    ):
        super().__init__(
            model_dir=model_dir,
            seed=seed,
            user_name=user_name,
            job_id=job_id,
            gpu=gpu,
            log_save_steps=log_save_steps,
            summary_save_steps=summary_save_steps,
            profiler_save_secs=profiler_save_secs,
            checkpoint_save_secs=checkpoint_save_secs,
            logger=logger,
        )
        assert trainer == TrainerType.PS

        self.task_index = task_index
        self.ps_hosts = ps_hosts
        self.job_name = job_name
        self.worker_hosts = worker_hosts
        self.worker_size = len(worker_hosts)

        self.tfserver, self.cluster = None, None
        if self.__is_worker() or self.__is_parameter_server():
            ## distributed job will set job_name.
            self.tfserver, self.cluster = self.__get_dist_training_server()

        if self.__is_parameter_server():
            ## parameter server will join here and never exits.
            tf.compat.v1.logging.info(
                "parameter servier {}-{} starts".format(self.job_name, self.task_index)
            )
            self.__ps_join()

        ## MonitoredTrainingSession parameters.
        self.is_chief = self.task_index == 0
        self.session_target = self.tfserver.target if self.tfserver is not None else ""
        self.checkpoint_dir = self.model_dir
        self.config = tf.compat.v1.ConfigProto()
        self.config.gpu_options.allow_growth = True

        self.dist_sync = DistributedSync(
            self.model_dir, self.task_index, len(self.worker_hosts)
        )

        self.parameter_server_num = len(self.ps_hosts)  # type: ignore

    def tf_device(self):
        return tf.compat.v1.device(
            tf.compat.v1.train.replica_device_setter(
                cluster=self.cluster,
                worker_device="/job:worker/task:{}".format(self.task_index),
            )
        )

    def __is_parameter_server(self):
        return self.job_name == "ps"

    def __is_worker(self):
        return self.job_name == "worker"

    def __ps_join(self):
        assert self.job_name == "ps"
        self.tfserver.join()

    def __get_dist_training_server(self):
        assert len(self.ps_hosts) > 0
        assert len(self.worker_hosts) > 0
        clustersepc = tf.train.ClusterSpec(
            {"ps": self.ps_hosts, "worker": self.worker_hosts}
        )
        server = tf.distribute.Server(
            clustersepc, job_name=self.job_name, task_index=self.task_index
        )
        return server, clustersepc
