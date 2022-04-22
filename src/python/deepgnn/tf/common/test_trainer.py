# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import tempfile, os, glob
import numpy as np
from unittest.mock import Mock
from deepgnn.tf.common.trainer import BaseTFTrainer
from deepgnn.tf.common.ps_trainer import PSTrainer
from deepgnn.tf.common.args import TrainerType
from deepgnn.tf.common.utils import node_embedding_to_string, reset_tf_graph
from deepgnn.tf.common.dataset import create_tf_dataset
from deepgnn.graph_engine.samplers import RangeNodeSampler
import tensorflow as tf


def test_trainer_query_fn():
    trainer = BaseTFTrainer(model_dir="", seed=0)

    trainer._train_impl = Mock()

    def test_case_default_model_query():
        ## test case: default model.query
        model = Mock()
        v = np.array([-1], np.int64)
        model.query.return_value = list([v]), list([v.shape])
        tf_dataset, _ = create_tf_dataset(
            sampler_class=RangeNodeSampler,
            query_fn=model.query,
            backend_options=None,
            first=0,
            last=100,
            batch_size=2,
            worker_index=0,
            num_workers=1,
            backfill_id=-1,
            prefetch_worker_size=1,
            prefetch_queue_size=2,
        )

        trainer.train(dataset=tf_dataset, model=model, optimizer=None)
        model.query.assert_called()

    test_case_default_model_query()

    def test_case_user_query_fn():
        # test case: use query_fn
        model = Mock()
        query_fn = Mock()
        v = np.array([-1], np.int64)
        query_fn.return_value = list([v]), list([v.shape])
        tf_dataset, _ = create_tf_dataset(
            sampler_class=RangeNodeSampler,
            query_fn=query_fn,
            backend_options=None,
            first=0,
            last=100,
            batch_size=2,
            worker_index=0,
            num_workers=1,
            backfill_id=-1,
            prefetch_worker_size=1,
            prefetch_queue_size=2,
        )

        trainer.train(dataset=tf_dataset, model=model, optimizer=None)
        query_fn.assert_called()
        model.query.assert_not_called()

    test_case_user_query_fn()


def test_trainer_query_fn_eval():
    trainer = BaseTFTrainer(model_dir="", seed=0)
    trainer._eval_impl = Mock()

    def test_case_default_model_query():
        ## test case: default model.query
        model = Mock()
        v = np.array([-1], np.int64)
        model.query.return_value = list([v]), list([v.shape])
        tf_dataset, _ = create_tf_dataset(
            sampler_class=RangeNodeSampler,
            query_fn=model.query,
            backend_options=None,
            first=0,
            last=100,
            batch_size=2,
            worker_index=0,
            num_workers=1,
            backfill_id=-1,
            prefetch_worker_size=1,
            prefetch_queue_size=2,
        )

        trainer.evaluate(dataset=tf_dataset, model=model)
        model.query.assert_called()

    test_case_default_model_query()

    def test_case_user_query_fn():
        # test case: use query_fn
        model = Mock()
        query_fn = Mock()
        v = np.array([-1], np.int64)
        query_fn.return_value = list([v]), list([v.shape])
        tf_dataset, _ = create_tf_dataset(
            sampler_class=RangeNodeSampler,
            query_fn=query_fn,
            backend_options=None,
            first=0,
            last=100,
            batch_size=2,
            worker_index=0,
            num_workers=1,
            backfill_id=-1,
            prefetch_worker_size=1,
            prefetch_queue_size=2,
        )
        trainer.evaluate(dataset=tf_dataset, model=model)
        query_fn.assert_called()
        model.query.assert_not_called()

    test_case_user_query_fn()


class MockModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.w = tf.keras.layers.Dense(1, use_bias=False)
        self.called_count = 0

    def query(self, graph, inputs, return_shape=False):
        self.called_count += 1
        batch_size = inputs.shape[0]
        dim = 10
        feature = np.random.rand(batch_size, dim).astype(np.float32)
        g_tensors = (inputs, feature)
        if return_shape:
            shapes = [x.shape for x in g_tensors]
            return g_tensors, shapes
        return g_tensors

    def call(self, inputs, training=False):
        node_id, feat = inputs
        y = self.w(feat)
        self.loss = tf.reduce_sum(input_tensor=y)
        self.node_id = node_id
        self.feature = feat
        return None, self.loss, {}

    def predict_step(self, data: dict):
        """override base predict_step."""
        self(data, training=False)
        return [self.node_id, self.feature]


def test_trainer_train():
    reset_tf_graph()
    tmp_dir = tempfile.TemporaryDirectory()

    trainer = PSTrainer(
        model_dir=tmp_dir.name,
        task_index=0,
        job_name=None,
        ps_hosts=[],
        worker_hosts=[],
        trainer=TrainerType.PS,
        log_save_steps=10,
        summary_save_steps=10,
        profiler_save_secs=180,
        checkpoint_save_secs=1,
        seed=None,
    )
    model = MockModel()
    tf_dataset, steps = create_tf_dataset(
        sampler_class=RangeNodeSampler,
        query_fn=model.query,
        backend_options=None,
        first=0,
        last=100,
        batch_size=2,
        worker_index=0,
        num_workers=1,
        backfill_id=-1,
        prefetch_worker_size=1,
        prefetch_queue_size=2,
    )
    trainer.train(
        dataset=tf_dataset,
        model=model,
        optimizer=tf.compat.v1.train.AdamOptimizer(0.01),
    )
    assert model.called_count == 51

    reset_tf_graph()
    global_step = tf.compat.v1.train.get_or_create_global_step()
    with tf.compat.v1.train.MonitoredTrainingSession(
        checkpoint_dir=tmp_dir.name,
        save_checkpoint_secs=None,
        save_checkpoint_steps=None,
    ) as sess:
        gstep_value = sess.run(global_step)
        assert gstep_value == steps

    tmp_dir.cleanup()


def test_trainer_eval():
    reset_tf_graph()
    tmp_dir = tempfile.TemporaryDirectory()

    trainer = PSTrainer(
        model_dir=tmp_dir.name,
        task_index=0,
        job_name=None,
        ps_hosts=[],
        worker_hosts=[],
        trainer=TrainerType.PS,
        log_save_steps=10,
        summary_save_steps=10,
        profiler_save_secs=180,
        checkpoint_save_secs=1,
        seed=None,
    )
    model = MockModel()
    tf_dataset, _ = create_tf_dataset(
        sampler_class=RangeNodeSampler,
        query_fn=model.query,
        backend_options=None,
        first=0,
        last=100,
        batch_size=2,
        worker_index=0,
        num_workers=1,
        backfill_id=-1,
        prefetch_worker_size=1,
        prefetch_queue_size=2,
    )
    trainer.evaluate(dataset=tf_dataset, model=model)
    assert model.called_count == 51

    tmp_dir.cleanup()


def test_trainer_inf():
    reset_tf_graph()
    tmp_dir = tempfile.TemporaryDirectory()

    trainer = PSTrainer(
        model_dir=tmp_dir.name,
        task_index=0,
        job_name=None,
        ps_hosts=[],
        worker_hosts=[],
        trainer=TrainerType.PS,
        log_save_steps=10,
        summary_save_steps=10,
        profiler_save_secs=180,
        checkpoint_save_secs=1,
        seed=None,
    )
    model = MockModel()
    tf_dataset, _ = create_tf_dataset(
        sampler_class=RangeNodeSampler,
        query_fn=model.query,
        backend_options=None,
        first=0,
        last=100,
        batch_size=2,
        worker_index=0,
        num_workers=1,
        backfill_id=-1,
        prefetch_worker_size=1,
        prefetch_queue_size=2,
    )
    trainer.inference(
        dataset=tf_dataset, model=model, embedding_to_str_fn=node_embedding_to_string
    )
    assert model.called_count == 51

    def check_embeddings(model_dir, num_nodes, dim):
        res = np.empty((num_nodes, dim), dtype=np.float32)
        files = glob.glob(os.path.join(model_dir, "embedding_*.tsv"))
        for fname in files:
            for line in open(fname):
                col = line.split("\t")
                nid = int(col[0])
                emb = [float(c) for c in col[1].split(" ")]
                assert len(emb) == dim
                res[nid] = emb
        return res

    check_embeddings(tmp_dir.name, 100, 10)
    tmp_dir.cleanup()
