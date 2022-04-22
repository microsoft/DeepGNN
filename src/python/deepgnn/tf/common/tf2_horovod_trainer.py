# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import time
import tensorflow as tf
import logging
import os

from typing import List
from tensorflow import keras
from deepgnn.tf.common.tf2_trainer import EagerTrainer
from deepgnn.tf.common.args import TrainerType

import horovod.tensorflow.keras as hvd


class HorovodEagerTrainer(EagerTrainer):
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
        checkpoint_save_secs: int = 3600,
        profile_batch: List[int] = [100, 100],
        logger: logging.Logger = None,
    ):
        hvd.init()
        super().__init__(
            model_dir=model_dir,
            seed=seed,
            user_name=user_name,
            job_id=job_id,
            gpu=gpu,
            log_save_steps=log_save_steps,
            summary_save_steps=summary_save_steps,
            checkpoint_save_secs=checkpoint_save_secs,
            profile_batch=profile_batch,
            logger=logger,
        )
        assert trainer == TrainerType.HVD
        self.logger.info(f"trainer - {trainer}")

        self.task_index = hvd.rank()
        self.worker_size = hvd.size()
        self.lr_scaler = hvd.size()

    def _set_gpu_device(self):
        if self.gpu:
            gpus = tf.config.experimental.list_physical_devices("GPU")
            self.logger.info(f"all GPU devices: {gpus}")
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            assert len(gpus) > hvd.local_rank()
            if gpus:
                tf.config.experimental.set_visible_devices(
                    gpus[hvd.local_rank()], "GPU"
                )
        else:
            tf.config.set_visible_devices([], "GPU")
        visible_devices = tf.config.get_visible_devices()
        for device in visible_devices:
            self.logger.info(f"device: {device}")

    def _train_impl(
        self, model, graph_dataset, optimizer, steps, loss, metrics, callbacks, epochs
    ):
        writer = self._get_summary_writer("train")
        s1 = time.time()
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

        def _log_fn(batch, logs):
            self._save_summary(writer, batch, logs)

        batch_callback = tf.keras.callbacks.LambdaCallback(
            on_batch_end=_log_fn, on_train_end=lambda _: writer.close()
        )

        cbs = [hvd.callbacks.BroadcastGlobalVariablesCallback(0), batch_callback]
        if callbacks:
            cbs.extend(callbacks)

        if hvd.rank() == 0:
            tb_callback = tf.keras.callbacks.TensorBoard(
                os.path.join(self.model_dir, "tboard")
            )
            cbs.append(keras.callbacks.ModelCheckpoint(self.checkpoint_file))
            cbs.append(tb_callback)

        model.fit(
            graph_dataset,
            epochs=epochs,
            callbacks=cbs,
            verbose=0,
            steps_per_epoch=steps,
        )
        self.logger.info(f"training time: {time.time() - s1}")
