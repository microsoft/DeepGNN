# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import time
import tensorflow as tf
import logging
import numpy as np
import random
import json
from typing import List, Union, Callable
from deepgnn.tf.common.utils import log_model_info
from deepgnn.tf.common.utils import node_embedding_to_string
from deepgnn.tf.common.base_trainer import Trainer
from deepgnn import (
    LOG_PROPS_EVENT_START_WORKER,
    LOG_PROPS_EVENT_END_WORKER,
    LOG_PROPS_PLATFORM_TF,
    log_telemetry,
)


class Stopwatch:
    def __init__(self):
        self._prev_time = time.time()

    def tictoc(self):
        current = time.time()
        delta = current - self._prev_time
        self._prev_time = current
        return delta


class LoggerCallback(tf.keras.callbacks.Callback):
    def __init__(
        self,
        name: str = "train",
        logger: logging.Logger = None,
        log_save_steps: int = 20,
    ):
        super().__init__()
        self._name = name
        self._log_save_steps = log_save_steps
        self._logger = logger
        self._timer = Stopwatch()
        self._first = True
        self._steps = 0
        self._epoch = 0

    def on_batch_end(self, batch, logs=None):
        self._log_step(logs)
        self._steps += 1

    def on_test_batch_end(self, batch, logs=None):
        self._log_step(logs)
        self._steps += 1

    def on_predict_batch_end(self, batch, logs=None):
        raise NotImplementedError

    def on_epoch_begin(self, epoch, logs=None):
        self._epoch = epoch

    def _log_step(self, logs):
        if self._first:
            self._first = False
            log_model_info(self.model, use_tf_compat=False)

        if self._steps % self._log_save_steps != 0:
            return

        output = ", ".join([f"{x}:{y:.4f}" for x, y in logs.items()])
        sec_per_step = self._timer.tictoc() / self._log_save_steps
        self._logger.info(
            f"{self._name}-step {self._steps}(epoch={self._epoch})  {output} ({sec_per_step:.4f} sec/step)"
        )


class EagerTrainer(Trainer):
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
        profile_batch: List[int] = [100, 100],
        logger: logging.Logger = None,
    ):
        task_index = 0
        worker_size = 1
        tf_config_str = os.environ.get("TF_CONFIG")
        if tf_config_str:
            tf_config = json.loads(tf_config_str)
            worker_size = len(tf_config["cluster"]["worker"])
            task_index = tf_config["task"]["index"]

        super().__init__(
            model_dir=model_dir,
            seed=seed,
            user_name=user_name,
            job_id=job_id,
            gpu=gpu,
            task_index=task_index,
            worker_size=worker_size,
            log_save_steps=log_save_steps,
            summary_save_steps=summary_save_steps,
            checkpoint_save_secs=checkpoint_save_secs,
            logger=logger,
        )

        self.lr_scaler = 1
        self.checkpoint_file = os.path.join(self.model_dir, "ckpt")

        self.set_random_seed(seed)
        self._set_gpu_device()

        self._ckpt_time = time.time()  # time info for checkpoint manager.

        self.profile_batch = ",".join([str(i) for i in profile_batch])

    def set_random_seed(self, seed: int = None):
        if seed:
            tf.random.set_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

    def _set_gpu_device(self):
        if self.gpu:
            gpus = tf.config.experimental.list_physical_devices("GPU")
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        else:
            tf.config.set_visible_devices([], "GPU")
        visible_devices = tf.config.get_visible_devices()
        for device in visible_devices:
            self.logger.info(f"device: {device}")

    def train(
        self,
        dataset: tf.data.Dataset,
        model: tf.keras.Model,
        optimizer: tf.keras.optimizers.Optimizer,
        loss: Union[str, Callable, tf.keras.losses.Loss] = None,
        metrics: List[Union[str, Callable, tf.keras.metrics.Metric]] = None,
        callbacks: List[tf.keras.callbacks.Callback] = None,
        epochs: int = 1,
        steps_per_epoch: int = None,
    ):
        """
        Args:
        * dataset: tf.data.Dataset which is used to get training batches.
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

        mode = "training"
        log_telemetry(
            self.logger,
            f"Training worker started.",
            LOG_PROPS_EVENT_START_WORKER,
            mode,
            type(model).__name__,
            self.user_name,
            self.job_id,
            self.task_index,
            self.worker_size,
            LOG_PROPS_PLATFORM_TF,
        )

        self.logger.info(
            f"training num_steps/epochs: {steps_per_epoch}, num_epochs: {epochs}"
        )

        model.optimizer = optimizer
        self._train_impl(
            model, dataset, optimizer, steps_per_epoch, loss, metrics, callbacks, epochs
        )

        log_telemetry(
            self.logger,
            f"Training worker finished.",
            LOG_PROPS_EVENT_END_WORKER,
            mode,
            type(model).__name__,
            self.user_name,
            self.job_id,
            self.task_index,
            self.worker_size,
            LOG_PROPS_PLATFORM_TF,
        )

    def _get_summary_writer(self, name):
        summary_dir = os.path.join(self.model_dir, name, f"worker_{self.task_index}")
        writer = tf.summary.create_file_writer(summary_dir)
        return writer

    def _save_summary(self, writer, step, logs=None):
        if step % self.summary_save_steps == 0:
            with writer.as_default():
                for name, value in logs.items():
                    tf.summary.scalar(name, value, step=step)

    def _save_checkpoint(self, checkpoint_manager, step, force=False):
        if self.task_index == 0:
            if force or time.time() - self._ckpt_time > self.checkpoint_save_secs:
                save_path = checkpoint_manager.save(step, check_interval=False)
                self.logger.info(
                    "Saved checkpoint step {}: {}".format(int(step), save_path)
                )
                self._ckpt_time = time.time()

    def _train_impl(
        self,
        model,
        graph_dataset,
        optimizer,
        steps,
        loss: Union[str, Callable, tf.keras.losses.Loss] = None,
        metrics: List[Union[str, Callable, tf.keras.metrics.Metric]] = None,
        callbacks: List[tf.keras.callbacks.Callback] = None,
        epochs: int = 1,
    ):
        timer = Stopwatch()

        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(self.checkpoint_file)

        log_callback = LoggerCallback("train", self.logger, self.log_save_steps)
        log_callback.set_model(model)
        tb_callback = tf.keras.callbacks.TensorBoard(
            log_dir=os.path.join(self.model_dir, "tboard"),
            profile_batch=self.profile_batch,
            update_freq=self.summary_save_steps,
        )

        cbs = [checkpoint_callback, log_callback, tb_callback]
        if callbacks:
            cbs.extend(callbacks)

        model.fit(
            graph_dataset,
            epochs=epochs,
            callbacks=cbs,
            verbose=0,
            steps_per_epoch=steps,
        )
        self.logger.info(f"training time: {timer.tictoc()}")

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

        mode = "evaluate"
        log_telemetry(
            self.logger,
            f"Training worker started.",
            LOG_PROPS_EVENT_START_WORKER,
            mode,
            type(model).__name__,
            self.user_name,
            self.job_id,
            self.task_index,
            self.worker_size,
            LOG_PROPS_PLATFORM_TF,
        )

        self.logger.info("evaluate num_steps: {}".format(steps))
        s = time.time()
        self._eval_impl(model, dataset, steps, loss, metrics, callbacks)
        self.logger.info(f"evaluate time: {time.time() - s}")

        log_telemetry(
            self.logger,
            f"Training worker finished.",
            LOG_PROPS_EVENT_END_WORKER,
            mode,
            type(model).__name__,
            self.user_name,
            self.job_id,
            self.task_index,
            self.worker_size,
            LOG_PROPS_PLATFORM_TF,
        )

    def _eval_impl(self, model, graph_dataset, steps, loss, metrics, callbacks):
        writer = self._get_summary_writer("evaluate")
        model.compile(loss=loss, metrics=metrics)

        def _log_fn(batch, logs):
            if batch == 0:
                log_model_info(model, use_tf_compat=False)
            self._save_summary(writer, batch, logs)

        log_callback = LoggerCallback("eval", self.logger, self.log_save_steps)
        log_callback.set_model(model)
        batch_callback = tf.keras.callbacks.LambdaCallback(
            on_test_batch_end=_log_fn, on_test_end=lambda _: writer.close()
        )
        tb_callback = tf.keras.callbacks.TensorBoard(
            log_dir=os.path.join(self.model_dir, "tboard"),
            profile_batch=self.profile_batch,
        )

        cbs = [batch_callback, tb_callback, log_callback]
        if callbacks:
            cbs.extend(callbacks)

        model.load_weights(self.checkpoint_file)
        model.evaluate(graph_dataset, callbacks=cbs, verbose=0, steps=steps)

    def inference(
        self,
        dataset: tf.data.Dataset,
        model: tf.keras.Model,
        embedding_to_str_fn: Callable = None,
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

        if embedding_to_str_fn is None:
            embedding_to_str_fn = node_embedding_to_string

        mode = "inference"
        log_telemetry(
            self.logger,
            f"Training worker started.",
            LOG_PROPS_EVENT_START_WORKER,
            mode,
            type(model).__name__,
            self.user_name,
            self.job_id,
            self.task_index,
            self.worker_size,
            LOG_PROPS_PLATFORM_TF,
        )

        self.logger.info("inference num_steps: {}".format(steps))
        s = time.time()
        self._inference_impl(
            model, dataset, steps, embedding_to_str_fn, output_embedding_file_prefix
        )
        self.logger.info(f"inference time: {time.time() - s}")

        log_telemetry(
            self.logger,
            f"Training worker finished.",
            LOG_PROPS_EVENT_END_WORKER,
            mode,
            type(model).__name__,
            self.user_name,
            self.job_id,
            self.task_index,
            self.worker_size,
            LOG_PROPS_PLATFORM_TF,
        )

    def _inference_impl(
        self,
        model,
        graph_dataset,
        steps,
        embedding_to_str_fn,
        output_embedding_file_prefix,
    ):
        fname = os.path.join(
            self.model_dir,
            "{}_{}.tsv".format(output_embedding_file_prefix, self.task_index),
        )
        self.logger.info("embedding file: {0}".format(fname))

        model.load_weights(self.checkpoint_file)
        with tf.io.gfile.GFile(fname, "w") as fout:

            def _log_fn(batch, logs):
                if batch == 0:
                    log_model_info(model, use_tf_compat=False)
                embedding_list = logs["outputs"]
                output = embedding_to_str_fn(embedding_list)
                for line in output:
                    fout.write(line)

            batch_callback = tf.keras.callbacks.LambdaCallback(
                on_predict_batch_end=_log_fn
            )
            tb_callback = tf.keras.callbacks.TensorBoard(
                log_dir=os.path.join(self.model_dir, "tboard"),
                profile_batch=self.profile_batch,
                update_freq=self.summary_save_steps,
            )

            model.predict(
                graph_dataset,
                callbacks=[batch_callback, tb_callback],
                verbose=0,
                steps=steps,
            )
