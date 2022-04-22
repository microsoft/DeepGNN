# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import time
import logging
import tensorflow as tf
import numpy as np
import random

from typing import List, Union, Callable
from deepgnn.tf.common.utils import (
    setup_worker_hooks,
    setup_chief_only_hooks,
    log_model_info,
)
from deepgnn.tf.common.dist_sync import DistributedSync
from deepgnn.tf.common.base_trainer import Trainer
from deepgnn import (
    LOG_PROPS_EVENT_START_WORKER,
    LOG_PROPS_EVENT_END_WORKER,
    LOG_PROPS_PLATFORM_TF,
    log_telemetry,
)


tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
tf.compat.v1.disable_v2_behavior()


class BaseTFTrainer(Trainer):
    def __init__(
        self,
        model_dir: str,
        seed: int,
        user_name: str = "",
        job_id: str = "",
        gpu: bool = False,
        log_save_steps: int = 20,
        summary_save_steps: int = 100,
        checkpoint_save_secs: int = 3600,
        profiler_save_secs: int = 180,
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
            checkpoint_save_secs=checkpoint_save_secs,
            logger=logger,
        )

        self.lr_scaler = 1
        self.parameter_server_num = None

        ## MonitoredTrainingSession parameters.
        self.checkpoint_dir = self.model_dir
        self.is_chief = True
        self.session_target = ""
        self.config: tf.compat.v1.ConfigProto = None

        self.dist_sync: DistributedSync = None  # type: ignore
        self.set_random_seed(self.seed)
        self.profiler_save_secs = profiler_save_secs

    def set_random_seed(self, seed: int = None):
        if seed:
            tf.compat.v1.set_random_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

    def tf_device(self):
        return tf.compat.v1.device(
            tf.compat.v1.train.replica_device_setter(cluster=None)
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
        Args:
        * dataset: tf.data.Dataset which is used to get subgraph.
        * model: `tf.keras.Model`, train() will call `tf.keras.Model.__call__()` to calcluate embedding, loss and metrics.
        * optimizer: training optimizer.
        * loss: loss for training model, String (name of objective function), objective function or tf.keras.losses.Loss instance.
            For TF1 models, loss is a Callable, return a tensor.
        * metrics: metrics for evaluation in _train_step, List of metrics to be evaluated by the model during training and testing.
            Each of this can be a string (name of a built-in function), function or a tf.keras.metrics.Metric instance.
        * callbacks: List of keras.callbacks.Callback instances.
        * epochs: Number of epochs to train the model.
        * steps_per_epoch: Number of steps to run in one epoch.
        """
        assert dataset is not None
        assert callbacks is None, "TF1 Trainer doesn't support callbacks"
        model.mode = "training"

        log_telemetry(
            self.logger,
            f"Training worker started.",
            LOG_PROPS_EVENT_START_WORKER,
            model.mode,
            type(model).__name__,
            self.user_name,
            self.job_id,
            self.task_index,
            self.worker_size,
            LOG_PROPS_PLATFORM_TF,
        )

        tf.compat.v1.logging.info("training num_steps: {}".format(steps_per_epoch))
        tf_context = self.make_dataset_iterator(dataset, epochs)
        self._train_impl(model, tf_context, loss, metrics, optimizer)
        if self.dist_sync:
            self.dist_sync.sync("final")

        log_telemetry(
            self.logger,
            f"Training worker finished.",
            LOG_PROPS_EVENT_END_WORKER,
            model.mode,
            type(model).__name__,
            self.user_name,
            self.job_id,
            self.task_index,
            self.worker_size,
            LOG_PROPS_PLATFORM_TF,
        )

    def _train_impl(self, model, context, loss, metrics, optimizer):
        _, loss_v, metrics_v = model(context, training=True)
        if loss is not None:
            loss_v = loss()
        assert type(loss_v) is tf.Tensor
        if metrics is not None:
            # for tf1 trainer, only 1 custom metrics function supported.
            assert len(metrics) == 1
            metrics_v = metrics[0]()
        assert type(metrics_v) is dict

        global_step = tf.compat.v1.train.get_or_create_global_step()
        train_op = optimizer.minimize(loss_v, global_step=global_step)

        tf_ops = [train_op]

        hooks, chiefhooks = self._setup_training_hooks(
            self.task_index,
            self.checkpoint_dir,
            global_step,
            loss_v,
            metrics_v,
            dist_sync=self.dist_sync,
        )

        ## debug log
        log_model_info(model)
        s1 = time.time()
        first_batch = True
        with tf.compat.v1.train.MonitoredTrainingSession(
            master=self.session_target,
            is_chief=self.is_chief,
            checkpoint_dir=self.checkpoint_dir,
            save_checkpoint_secs=None,
            save_checkpoint_steps=None,
            log_step_count_steps=None,
            hooks=hooks,
            chief_only_hooks=chiefhooks,
            config=self.config,
        ) as sess:
            while not sess.should_stop():
                sess.run(tf_ops)
                if first_batch:
                    s2 = time.time()
                    first_batch = False

        tf.compat.v1.logging.info(
            f"init time: {s2 - s1}, training time: {time.time() - s2}"
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
        Args:
        * dataset: tf.data.Dataset which is used to get subgraph.
        * model: `tf.keras.Model`, train() will call `tf.keras.Model.__call__()` to calcluate embedding, loss and metrics.
        * embedding_to_str_fn: convert a list of tensors to output string.
        * output_embedding_file_prefix: the embedding file will be {model_dir}/{prefix}_{task_index}.txt"
        * steps: Number of steps to run.
        """
        assert dataset is not None
        model.mode = "inference"

        log_telemetry(
            self.logger,
            f"Training worker started.",
            LOG_PROPS_EVENT_START_WORKER,
            model.mode,
            type(model).__name__,
            self.user_name,
            self.job_id,
            self.task_index,
            self.worker_size,
            LOG_PROPS_PLATFORM_TF,
        )

        tf.compat.v1.logging.info("inference num_steps: {}".format(steps))
        tf_context = self.make_dataset_iterator(dataset)
        self._inference_impl(
            model, tf_context, embedding_to_str_fn, output_embedding_file_prefix
        )
        if self.dist_sync:
            self.dist_sync.sync("final")

        log_telemetry(
            self.logger,
            f"Training worker finished.",
            LOG_PROPS_EVENT_END_WORKER,
            model.mode,
            type(model).__name__,
            self.user_name,
            self.job_id,
            self.task_index,
            self.worker_size,
            LOG_PROPS_PLATFORM_TF,
        )

    def _inference_impl(
        self, model, context, embedding_to_str_fn, output_embedding_file_prefix
    ):
        log_model_info(model)
        global_step = tf.compat.v1.train.get_or_create_global_step()
        add_step_op = tf.compat.v1.assign(
            global_step, global_step + 1, use_locking=True
        )

        emb_list = model.predict_step(context)
        assert type(emb_list) is list and len(emb_list) > 0
        assert type(emb_list[0]) is tf.Tensor

        hooks = self._setup_inference_hooks(
            self.task_index, global_step, self.dist_sync
        )
        fname = os.path.join(
            self.model_dir,
            "{}_{}.tsv".format(output_embedding_file_prefix, self.task_index),
        )
        tf.compat.v1.logging.info("embedding file: {0}".format(fname))

        with tf.io.gfile.GFile(fname, "w") as fout:
            with tf.compat.v1.train.MonitoredTrainingSession(
                master=self.session_target,
                is_chief=self.is_chief,
                checkpoint_dir=self.checkpoint_dir,
                save_checkpoint_secs=None,
                save_checkpoint_steps=None,
                log_step_count_steps=None,
                hooks=hooks,
                config=self.config,
            ) as sess:
                while not sess.should_stop():
                    res = sess.run(emb_list + [add_step_op])
                    embedding_tesor_list = res[0:-1]
                    output = embedding_to_str_fn(embedding_tesor_list)
                    for line in output:
                        fout.write(line)

        tf.compat.v1.logging.info("inference is finished.")

    def evaluate(
        self,
        dataset: tf.data.Dataset,
        model: tf.keras.Model,
        loss: Union[str, Callable, tf.keras.losses.Loss] = None,
        metrics: List[Union[str, Callable, tf.keras.metrics.Metric]] = None,
        callbacks: List[tf.keras.callbacks.Callback] = None,
        steps: int = None,
    ):
        """
        Args:
        * dataset: tf.data.Dataset which is used to get subgraph.
        * model: `tf.keras.Model`, evaluate() will call `tf.keras.Model.__call__()` to calcluate embedding, loss and metrics.
        * loss: loss for training model, String (name of objective function), objective function or tf.keras.losses.Loss instance.
            For TF1 models, loss is a Callable, return a tensor.
        * metrics: metrics for evaluation in _train_step, List of metrics to be evaluated by the model during training and testing.
            Each of this can be a string (name of a built-in function), function or a tf.keras.metrics.Metric instance.
        * callbacks: List of keras.callbacks.Callback instances.
        * steps: Number of steps to run.
        """
        assert dataset is not None
        model.mode = "evaluate"

        log_telemetry(
            self.logger,
            f"Training worker started.",
            LOG_PROPS_EVENT_START_WORKER,
            model.mode,
            type(model).__name__,
            self.user_name,
            self.job_id,
            self.task_index,
            self.worker_size,
            LOG_PROPS_PLATFORM_TF,
        )

        tf.compat.v1.logging.info("evaluate num_steps: {}".format(steps))
        tf_context = self.make_dataset_iterator(dataset)
        self._eval_impl(model, tf_context, loss, metrics)
        if self.dist_sync:
            self.dist_sync.sync("final")

        log_telemetry(
            self.logger,
            f"Training worker finished.",
            LOG_PROPS_EVENT_END_WORKER,
            model.mode,
            type(model).__name__,
            self.user_name,
            self.job_id,
            self.task_index,
            self.worker_size,
            LOG_PROPS_PLATFORM_TF,
        )

    def _eval_impl(self, model, context, loss, metrics):
        _, loss_v, metrics_v = model(context, training=True)
        if loss is not None:
            loss_v = loss()
        assert type(loss_v) is tf.Tensor
        if metrics is not None:
            # for tf1 trainer, only 1 custom metrics function supported.
            assert len(metrics) == 1
            metrics_v = metrics[0]()
        assert type(metrics_v) is dict

        global_step = tf.compat.v1.train.get_or_create_global_step()
        add_step_op = tf.compat.v1.assign(
            global_step, global_step + 1, use_locking=True
        )

        hooks = self._setup_eval_hooks(
            self.task_index, global_step, loss_v, metrics_v, dist_sync=self.dist_sync
        )

        if type(context) is tuple:
            tmp = context[0]
        elif type(context) is dict:
            tmp = context["inputs"]
        else:
            raise ValueError(f"context type: {type(context)}, {context}")
        tf_ops = [add_step_op, tmp]

        ## debug log
        log_model_info(model)
        with tf.compat.v1.train.MonitoredTrainingSession(
            master=self.session_target,
            is_chief=self.is_chief,
            checkpoint_dir=self.checkpoint_dir,
            save_checkpoint_secs=None,
            save_checkpoint_steps=None,
            log_step_count_steps=None,
            hooks=hooks,
            config=self.config,
        ) as sess:
            while not sess.should_stop():
                sess.run(tf_ops)

    def _setup_training_hooks(
        self, task_index, checkpoint_dir, global_step, loss, metrics, dist_sync=None
    ):
        logging_tensor = {"step": global_step, "loss": loss}
        logging_tensor.update(metrics)
        summary_tensor = {"loss": loss}
        summary_tensor.update(metrics)
        hooks = setup_worker_hooks(
            self.model_dir,
            task_index,
            logging_tensor=logging_tensor,
            summary_tensor=summary_tensor,
            dist_sync=dist_sync,
            metric_dir="train",
            log_save_steps=self.log_save_steps,
            summary_save_steps=self.summary_save_steps,
            profiler_save_secs=self.profiler_save_secs,
        )
        chiefhooks = setup_chief_only_hooks(
            task_index,
            self.checkpoint_dir,
            dist_sync,
            checkpoint_save_secs=self.checkpoint_save_secs,
        )
        return hooks, chiefhooks

    def _setup_inference_hooks(self, task_index, global_step, dist_sync=None):
        hooks = setup_worker_hooks(
            self.model_dir,
            task_index,
            logging_tensor={"inf_step": global_step},
            dist_sync=dist_sync,
            log_save_steps=self.log_save_steps,
            summary_save_steps=self.summary_save_steps,
            profiler_save_secs=self.profiler_save_secs,
        )
        return hooks

    def _setup_eval_hooks(self, task_index, global_step, loss, metrics, dist_sync=None):
        logging_tensor = {"eval_step": global_step, "loss": loss}
        logging_tensor.update(metrics)
        summary_tensor = {"loss": loss}
        summary_tensor.update(metrics)
        hooks = setup_worker_hooks(
            self.model_dir,
            task_index,
            logging_tensor=logging_tensor,
            summary_tensor=summary_tensor,
            dist_sync=dist_sync,
            metric_dir="evaluate",
            log_save_steps=self.log_save_steps,
            summary_save_steps=self.summary_save_steps,
            profiler_save_secs=self.profiler_save_secs,
        )
        return hooks

    def make_dataset_iterator(self, graph_dataset, epochs: int = 1):
        if epochs > 1:
            graph_dataset = graph_dataset.repeat(epochs)
        source = tf.compat.v1.data.make_one_shot_iterator(graph_dataset).get_next()
        return source
