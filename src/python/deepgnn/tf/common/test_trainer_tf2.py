# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import tempfile, os
import numpy as np
from unittest.mock import Mock
from deepgnn.tf.common import trainer_factory
from deepgnn.tf.common.args import TrainerType
from deepgnn.tf.common.test_helper import TestHelper
from deepgnn.tf.common.dataset import create_tf_dataset
from deepgnn.graph_engine.samplers import FileNodeSampler, TextFileSampler
from deepgnn.graph_engine import Graph
import tensorflow as tf


class ModelQuery:
    def query(self, graph: Graph, inputs: np.array, return_shape: bool = False):
        if isinstance(inputs, list):
            inputs = np.array(inputs, dtype=np.int64)

        x = inputs.reshape(-1, 1)
        bias = np.random.rand(x.size) / 10.0
        y = (2 * x + 3.0) + bias

        x = x.astype(np.float32)
        y = y.astype(np.float32)

        tensors = (x, y)
        if return_shape:
            return tensors, (x.shape, y.shape)
        else:
            return tensors


class TestModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.a = tf.Variable(np.random.randn(), name="a")
        self.b = tf.Variable(np.random.randn(), name="b")

    def call(self, inputs, training=False):
        x, y = inputs
        y_pred = self.a * x + self.b
        self.loss = tf.reduce_sum(tf.keras.losses.mean_squared_error(y, y_pred))
        return self.loss, {"loss_v": self.loss}

    def dump(self):
        return self.dump_value

    def train_step(self, data: dict):
        """override base train_step."""
        with tf.GradientTape() as tape:
            loss, metrics = self(data, training=True)

        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        result = {"loss": loss}
        result.update(metrics)
        return result

    def test_step(self, data: dict):
        """override base test_step."""
        loss, metrics = self(data, training=False)
        result = {"loss": loss}
        result.update(metrics)
        return result

    def predict_step(self, data: dict):
        """override base predict_step."""
        self(data, training=False)
        return [self.node_id, self.feature]


def get_param(model_dir):
    param = Mock()
    param.eager = True
    param.trainer = TrainerType.PS
    param.learning_rate = 0.00001
    param.prefetch_queue_size = 1
    param.prefetch_worker_size = 1
    param.model_dir = model_dir
    param.task_index = 0
    param.ps_hosts = []
    param.worker_hosts = []
    param.log_save_steps = 2
    param.summary_save_steps = 1
    param.profiler_save_secs = 180
    param.checkpoint_save_secs = 1
    param.seed = 123
    param.gpu = False
    param.profile_batch = [2, 5]
    return param


def run_job(trainer_type):
    tmp_dir = tempfile.TemporaryDirectory()
    param = get_param(tmp_dir.name)
    param.trainer = trainer_type

    fname = os.path.join(tmp_dir.name, "x.txt")
    with open(fname, "w") as fout:
        for i in range(100):
            fout.write(f"{i}\n")

    trainer = trainer_factory.get_trainer(param)
    q = ModelQuery()
    model = TestModel()
    tf_dataset, _ = create_tf_dataset(
        sampler_class=FileNodeSampler,
        query_fn=q.query,
        backend_options=None,
        sample_files=fname,
        batch_size=20,
        shuffle=True,
        drop_last=True,
        prefetch_worker_size=1,
        prefetch_queue_size=2,
    )

    trainer.train(
        dataset=tf_dataset,
        model=model,
        optimizer=tf.keras.optimizers.SGD(param.learning_rate),
        epochs=5,
    )

    model = TestModel()
    tf_dataset, _ = create_tf_dataset(
        sampler_class=FileNodeSampler,
        query_fn=q.query,
        backend_options=None,
        sample_files=fname,
        batch_size=10,
        shuffle=True,
        drop_last=True,
        prefetch_worker_size=1,
        prefetch_queue_size=2,
    )

    trainer.evaluate(dataset=tf_dataset, model=model)

    loss_v = TestHelper.get_tf2_summary_value(
        os.path.join(param.model_dir, "evaluate/worker_0"), "loss_v"
    )
    avg_loss = sum(loss_v) / len(loss_v)
    assert avg_loss < 10.418


def test_eager_trainer():
    # fix pytest for tf1|tf2 trainer: install pytest-xdist.
    # reference: https://stackoverflow.com/questions/48234032/run-py-test-test-in-different-process
    tf.compat.v1.enable_v2_behavior()
    run_job(TrainerType.PS)
    run_job(TrainerType.HVD)


def test_eager_trainer_with_mirrored_strategy():
    tf.compat.v1.enable_v2_behavior()
    with tf.distribute.MirroredStrategy().scope():
        run_job(TrainerType.PS)
        run_job(TrainerType.HVD)


def test_eager_trainer_with_default_strategy():
    tf.compat.v1.enable_v2_behavior()
    with tf.distribute.get_strategy().scope():
        run_job(TrainerType.PS)
        run_job(TrainerType.HVD)


def test_distributed_dataset_with_no_fixed_length():
    tf.compat.v1.enable_v2_behavior()
    tmp_dir = tempfile.TemporaryDirectory()
    param = get_param(tmp_dir.name)
    param.trainer = TrainerType.PS

    fname = os.path.join(tmp_dir.name, "x.txt")
    with open(fname, "w") as fout:
        for i in range(100):
            fout.write(f"{i}\n")

    trainer = trainer_factory.get_trainer(param)

    with tf.distribute.MirroredStrategy().scope():
        q = ModelQuery()
        model = TestModel()
        tf_dataset, _ = create_tf_dataset(
            sampler_class=TextFileSampler,
            query_fn=q.query,
            backend_options=None,
            batch_size=20,
            prefetch_worker_size=1,
            prefetch_queue_size=2,
            store_name=None,
            filename=fname,
            shuffle=False,
            drop_last=False,
            worker_index=0,
            num_workers=1,
        )

        distributed_dataset = (
            tf.distribute.get_strategy().distribute_datasets_from_function(
                lambda ctx: tf_dataset.repeat(5)
            )
        )

        trainer.train(
            dataset=distributed_dataset,
            model=model,
            optimizer=tf.keras.optimizers.SGD(param.learning_rate),
            epochs=5,
            steps_per_epoch=5,
        )

    model = TestModel()
    tf_dataset, _ = create_tf_dataset(
        sampler_class=FileNodeSampler,
        query_fn=q.query,
        backend_options=None,
        sample_files=fname,
        batch_size=10,
        shuffle=True,
        drop_last=True,
        prefetch_worker_size=1,
        prefetch_queue_size=2,
    )

    trainer.evaluate(dataset=tf_dataset, model=model)

    loss_v = TestHelper.get_tf2_summary_value(
        os.path.join(param.model_dir, "evaluate/worker_0"), "loss_v"
    )
    avg_loss = sum(loss_v) / len(loss_v)
    assert avg_loss < 10.418
