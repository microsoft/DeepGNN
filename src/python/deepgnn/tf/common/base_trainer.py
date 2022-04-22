# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import abc
import os
import tensorflow as tf
import logging

from typing import Callable, Union, List


class Trainer(abc.ABC):
    """Abstract class for TF1/TF2 trainers."""

    def __init__(
        self,
        model_dir: str,
        seed: int,
        task_index: int = 0,
        worker_size: int = 1,
        gpu: bool = False,
        user_name: str = "",
        job_id: str = "",
        log_save_steps: int = 20,
        summary_save_steps: int = 100,
        checkpoint_save_secs: int = 3600,
        logger: logging.Logger = None,
    ):
        # keras checkpoint callback will use the basename/dirname
        # to construct the full checkpoint path. here we add a "/"
        # to the end of the path.
        self.model_dir = os.path.join(model_dir, "./")
        self.seed = seed
        self.task_index = task_index
        self.worker_size = worker_size
        self.gpu = gpu
        self.user_name = user_name
        self.job_id = job_id
        self.log_save_steps = log_save_steps
        self.summary_save_steps = summary_save_steps
        self.checkpoint_save_secs = checkpoint_save_secs
        # get custom logger if not none, otherwise using root logger
        self.logger = logger if logger is not None else logging.getLogger()
        self.logger.info(f"tensorflow version: {tf.__version__}")

    @abc.abstractmethod
    def set_random_seed(self, seed: int = None):
        """Set random seed"""

    @abc.abstractmethod
    def train(
        self,
        dataset: tf.data.Dataset,
        model: tf.keras.Model,
        optimizer: Union[tf.keras.optimizers.Optimizer, tf.compat.v1.train.Optimizer],
        loss: Union[str, Callable, tf.keras.losses.Loss] = None,
        metrics: List[Union[str, Callable, tf.keras.metrics.Metric]] = None,
        callbacks: List[tf.keras.callbacks.Callback] = None,
        epochs: int = 1,
        steps_per_epoch: int = None,
    ):
        """Training interface.
        Args:
        * dataset: tf.data.Dataset which is used to get subgraph.
        * model: `tf.keras.Model`, train() will call `tf.keras.Model.__call__()` to calcluate embedding, loss and metrics.
        * optimizer: training optimizer.
        * loss: loss for training model, String (name of objective function), objective function or tf.keras.losses.Loss instance.
            For TF1 models, loss is a Callable, return a tensor.
        * metrics: metric_fn for evaluation in _train_step, List of metrics to be evaluated by the model during training and testing.
            Each of this can be a string (name of a built-in function), function or a tf.keras.metrics.Metric instance.
        * callbacks: List of keras.callbacks.Callback instances. For TF1 trainer callbacks are ignored, for TF2 trainer this callbacks
            will be passed to `model.fit` API. The trainer has its own default callback list, and this callbacks parameter provides
            an oppotunity for custom models to append their own callbacks.
        * epochs: Number of epochs to train the model.
        * steps_per_epoch: Number of steps to run in one epoch.
        """

    @abc.abstractmethod
    def inference(
        self,
        dataset: tf.data.Dataset,
        model: tf.keras.Model,
        embedding_to_str_fn: Callable,
        output_embedding_file_prefix: str = "embedding",
        steps: int = None,
    ):
        """Inference interface.
        Args:
        * dataset: tf.data.Dataset which is used to get subgraph.
        * model: `tf.keras.Model`, train() will call `tf.keras.Model.__call__()` to calcluate embedding, loss and metrics.
        * embedding_to_str_fn: convert a list of tensors to output string.
        * output_embedding_file_prefix: the embedding file will be {model_dir}/{prefix}_{task_index}.txt"
        * steps: Number of steps to run.
        """

    @abc.abstractmethod
    def evaluate(
        self,
        dataset: tf.data.Dataset,
        model: tf.keras.Model,
        loss: Union[str, Callable, tf.keras.losses.Loss] = None,
        metrics: List[Union[str, Callable, tf.keras.metrics.Metric]] = None,
        callbacks: List[tf.keras.callbacks.Callback] = None,
        steps: int = None,
    ):
        """Evaluation interface.
        Args:
        * dataset: tf.data.Dataset which is used to get subgraph.
        * model: `tf.keras.Model`, evaluate() will call `tf.keras.Model.__call__()` to calcluate embedding, loss and metrics.
        * loss: loss for training model, String (name of objective function), objective function or tf.keras.losses.Loss instance.
            For TF1 models, loss is a Callable, return a tensor.
        * metrics: metric_fn for evaluation in _train_step, List of metrics to be evaluated by the model during training and testing.
            Each of this can be a string (name of a built-in function), function or a tf.keras.metrics.Metric instance.
        * callbacks: List of keras.callbacks.Callback instances. For TF1 trainer callbacks are ignored, for TF2 trainer this callbacks
            will be passed to `model.fit` API. The trainer has its own default callback list, and this callbacks parameter provides
            an oppotunity for custom models to append their own callbacks.
        * steps: Number of steps to run.
        """
