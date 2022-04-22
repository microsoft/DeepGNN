# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import logging
import tensorflow as tf
from typing import List, Union, Callable

from deepgnn.tf.common.trainer import BaseTFTrainer
from deepgnn.tf.common.args import TrainerType
from deepgnn.tf.common.dist_sync import DistributedSync

import horovod.tensorflow as hvd


class HorovodTFTrainer(BaseTFTrainer):
    def __init__(
        self,
        trainer: TrainerType,
        model_dir: str,
        seed: int,
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
        assert trainer == TrainerType.HVD

        hvd.init()
        self.task_index = hvd.rank()
        self.worker_size = hvd.size()
        self.lr_scaler = hvd.size()

        ## Hovovod: tf.train.MonitoredTrainingSession: https://github.com/horovod/horovod/blob/master/docs/tensorflow.rst
        ## * is_chief: True
        ## * master(session_target): ""
        ## * checkpoint_dir: accomplish this by passing checkpoint_dir=None to tf.train.MonitoredTrainingSession if hvd.rank() != 0.
        ##    - training: DeepGNN use ChiefCheckpointSaverHook, rather than a default CheckpointSaverHook.
        ##    - evaluate/inference: DeepGNN set checkpoint_dir=None if hvd.rank() != 0.
        self.checkpoint_dir = (
            self.model_dir if self.task_index == 0 else None  # type: ignore
        )
        if self.gpu:
            self.config = tf.compat.v1.ConfigProto()
            self.config.gpu_options.allow_growth = True
            self.config.gpu_options.visible_device_list = str(hvd.local_rank())
        else:
            self.config = tf.compat.v1.ConfigProto(device_count={"GPU": 0})

        self.dist_sync = DistributedSync(
            self.model_dir, self.task_index, self.worker_size
        )

    def train(
        self,
        dataset: tf.data.Dataset,
        model: tf.keras.Model,
        optimizer: tf.compat.v1.train.Optimizer,
        loss: Union[str, Callable, tf.keras.losses.Loss] = None,
        metrics: List[Union[str, Callable, tf.keras.metrics.Metric]] = None,
        callbacks: List[tf.keras.callbacks.Callback] = None,
        epochs: int = 1,
        steps_per_epoch: int = None,
    ):
        """
        HorovodTFTrainer.train(), please override it if needed.
        * Wrap the optimizer in hvd.DistributedOptimizer.
        """
        hvd_optimizer = hvd.DistributedOptimizer(optimizer)
        super().train(
            dataset=dataset,
            model=model,
            loss=loss,
            metrics=metrics,
            optimizer=hvd_optimizer,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
        )

    def inference(
        self,
        dataset: tf.data.Dataset,
        model: tf.keras.Model,
        embedding_to_str_fn: Callable,
        output_embedding_file_prefix: str = "embedding",
        steps: int = None,
    ):
        """
        HorovodTFTrainer.inference(), please override it if needed.
        """
        super().inference(
            dataset, model, embedding_to_str_fn, output_embedding_file_prefix, steps
        )

    def evaluate(
        self,
        dataset: tf.data.Dataset,
        model: tf.keras.Model,
        loss: Union[str, Callable, tf.keras.losses.Loss] = None,
        metrics: List[Union[str, Callable, tf.keras.metrics.Metric]] = None,
        _: List[tf.keras.callbacks.Callback] = None,
        steps: int = None,
    ):
        """
        HorovodTFTrainer.evaluate(), please override it if needed.
        """
        super().evaluate(dataset, model, loss=loss, metrics=metrics, steps=steps)

    def _setup_training_hooks(
        self, task_index, checkpoint_dir, global_step, loss, metrics, dist_sync=None
    ):
        """
        Override BaseTFTrainer._setup_training_hooks().
        - add hvd.BroadcastGlobalVariablesHook(0) for model variables initliazation.
        """
        hooks, chiefhooks = super()._setup_training_hooks(
            self.task_index,
            self.checkpoint_dir,
            global_step,
            loss,
            metrics,
            self.dist_sync,
        )
        hooks.append(hvd.BroadcastGlobalVariablesHook(0))
        return hooks, chiefhooks

    def _setup_inference_hooks(self, task_index, global_step, dist_sync=None):
        """
        Override BaseTFTrainer._setup_inference_hooks().
        - add hvd.BroadcastGlobalVariablesHook(0) for model variables initliazation.
        """
        hooks = super()._setup_inference_hooks(
            self.task_index, global_step, dist_sync=self.dist_sync
        )
        hooks.append(hvd.BroadcastGlobalVariablesHook(0))
        return hooks

    def _setup_eval_hooks(self, task_index, global_step, loss, metrics, dist_sync=None):
        """
        Override BaseTFTrainer._setup_eval_hooks().
        - add hvd.BroadcastGlobalVariablesHook(0) for model variables initliazation.
        """
        hooks = super()._setup_eval_hooks(
            self.task_index, global_step, loss, metrics, self.dist_sync
        )
        hooks.append(hvd.BroadcastGlobalVariablesHook(0))
        return hooks
