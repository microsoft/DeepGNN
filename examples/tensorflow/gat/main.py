# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
import tensorflow as tf
import numpy as np

from deepgnn.graph_engine import (
    SamplingStrategy,
    GENodeSampler,
    RangeNodeSampler,
    FileNodeSampler,
    BackendOptions,
    create_backend,
)

from deepgnn import str2list_int, setup_default_logging_config
from contextlib import closing
from deepgnn.tf import common
from deepgnn.tf.common.dataset import create_tf_dataset, get_distributed_dataset
from deepgnn.tf.common.trainer_factory import get_trainer
from gat import GAT, GATQuery, GATQueryParameter  # type: ignore


# fmt: off
def define_param_gat(parser):
    parser.add_argument("--batch_size", type=int, default=16, help="mini-batch size")
    parser.add_argument("--epochs", type=int, default=200, help="num of epochs for training")
    parser.add_argument("--learning_rate", type=float, default=0.005, help="learning rate")

    # GAT Model Parameters.
    parser.add_argument("--head_num", type=str2list_int, default="8,1", help="the number of attention headers.")
    parser.add_argument("--hidden_dim", type=int, default=8, help="hidden layer dimension.")
    parser.add_argument("--num_classes", type=int, default=-1, help="number of classes for category")
    parser.add_argument("--ffd_drop", type=float, default=0.0, help="feature dropout rate.")
    parser.add_argument("--attn_drop", type=float, default=0.0, help="attention layer dropout rate.")
    parser.add_argument("--l2_coef", type=float, default=0.0005, help="l2 loss")

    # training node types.
    parser.add_argument("--node_types", type=str2list_int, default="0", help="Graph Node for training.")
    # evaluate node files.
    parser.add_argument("--evaluate_node_files", type=str, help="evaluate node file list.")
    # inference node id
    parser.add_argument("--inf_min_id", type=int, default=0, help="inferece min node id.")
    parser.add_argument("--inf_max_id", type=int, default=-1, help="inference max node id.")

    parser.add_argument(
        "--distributed_strategy",
        type=str,
        default=None,
        choices=[None, "Mirrored", "MultiWorkerMirrored"],
        help="Distributed strategies to use.",
    )

    def register_gat_query_param(parser):
        group = parser.add_argument_group("GAT Query Parameters")
        group.add_argument("--neighbor_edge_types", type=str2list_int, default="0", help="Graph Edge for attention encoder.",)
        group.add_argument("--feature_idx", type=int, default=0, help="feature index.")
        group.add_argument("--feature_dim", type=int, default=16, help="feature dim.")
        group.add_argument("--label_idx", type=int, default=1, help="label index.")
        group.add_argument("--label_dim", type=int, default=1, help="label dim.")

    register_gat_query_param(parser)
# fmt: on


def build_model(param):
    p = GATQueryParameter(
        neighbor_edge_types=np.array(param.neighbor_edge_types, np.int32),
        feature_idx=param.feature_idx,
        feature_dim=param.feature_dim,
        label_idx=param.label_idx,
        label_dim=param.label_dim,
        num_hops=len(param.head_num),
    )
    query_obj = GATQuery(p)

    model = GAT(
        head_num=param.head_num,
        hidden_dim=param.hidden_dim,
        num_classes=param.num_classes,
        ffd_drop=param.ffd_drop,
        attn_drop=param.attn_drop,
        l2_coef=param.l2_coef,
    )

    return model, query_obj


def run_train(param, trainer, query, model, tf1_mode, backend):
    tf_dataset, steps_per_epoch = create_tf_dataset(
        sampler_class=GENodeSampler,
        query_fn=query.query_training,
        backend=backend,
        node_types=np.array(param.node_types, dtype=np.int32),
        batch_size=param.batch_size,
        num_workers=trainer.worker_size,
        worker_index=trainer.task_index,
        strategy=SamplingStrategy.RandomWithoutReplacement,
    )

    distributed_dataset = get_distributed_dataset(
        # NOTE: here we flatten all the epochs into 1 to increase performance.
        lambda ctx: tf_dataset.repeat(param.epochs)
    )

    # we need to make sure the steps_per_epoch are provided in distributed dataset.
    assert steps_per_epoch is not None or param.steps_per_epoch is not None
    # Since we flatten the dataset to len(dataset) * param.epochs,
    # we alos need to update steps_per_epoch.
    steps_per_epoch = param.epochs * (steps_per_epoch or param.steps_per_epoch)

    if tf1_mode:
        opt = tf.compat.v1.train.AdamOptimizer(param.learning_rate * trainer.lr_scaler)
    else:
        opt = tf.keras.optimizers.Adam(
            learning_rate=param.learning_rate * trainer.lr_scaler
        )

    trainer.train(
        dataset=distributed_dataset,
        model=model,
        optimizer=opt,
        epochs=1,
        steps_per_epoch=steps_per_epoch,
    )


def run_eval(param, trainer, query, model, backend):
    tf_dataset, steps_per_epoch = create_tf_dataset(
        sampler_class=FileNodeSampler,
        query_fn=query.query_training,
        backend=backend,
        sample_files=param.evaluate_node_files,
        batch_size=param.batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=trainer.worker_size,
        worker_index=trainer.task_index,
    )

    distributed_dataset = get_distributed_dataset(lambda ctx: tf_dataset)
    # we need to make sure the steps_per_epoch are provided in distributed dataset.
    assert steps_per_epoch is not None or param.steps_per_epoch is not None

    trainer.evaluate(
        dataset=distributed_dataset,
        model=model,
        steps=(steps_per_epoch or param.steps_per_epoch),
    )


def run_inference(param, trainer, query, model, backend):
    tf_dataset, steps_per_epoch = create_tf_dataset(
        sampler_class=RangeNodeSampler,
        query_fn=query.query_training,
        backend=backend,
        first=param.inf_min_id,
        last=param.inf_max_id,
        batch_size=param.batch_size,
        num_workers=trainer.worker_size,
        worker_index=trainer.task_index,
        backfill_id=-1,
    )

    distributed_dataset = get_distributed_dataset(lambda ctx: tf_dataset)

    # we need to make sure the steps_per_epoch are provided in distributed dataset.
    assert steps_per_epoch is not None or param.steps_per_epoch is not None

    trainer.inference(
        dataset=distributed_dataset,
        model=model,
        steps=(steps_per_epoch or param.steps_per_epoch),
        embedding_to_str_fn=common.utils.node_embedding_to_string,
    )


def _main():
    # setup default logging component.
    setup_default_logging_config(enable_telemetry=True)

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter, allow_abbrev=False
    )
    common.args.import_default_parameters(parser)
    define_param_gat(parser)

    param = parser.parse_args()
    common.args.log_all_parameters(param)

    trainer = get_trainer(param)
    backend = create_backend(BackendOptions(param), is_leader=(trainer.task_index == 0))

    def run(tf1_mode=False):
        model, query = build_model(param)
        if param.mode == common.args.TrainMode.TRAIN:
            run_train(param, trainer, query, model, tf1_mode, backend)
        elif param.mode == common.args.TrainMode.EVALUATE:
            run_eval(param, trainer, query, model, backend)
        elif param.mode == common.args.TrainMode.INFERENCE:
            run_inference(param, trainer, query, model, backend)

    with closing(backend):
        if param.eager:
            strategy = None
            if param.distributed_strategy == "Default":
                strategy = tf.distribute.get_strategy()
            elif param.distributed_strategy == "Mirrored":
                strategy = tf.distribute.MirroredStrategy()
            elif param.distributed_strategy == "MultiWorkerMirrored":
                strategy = tf.distribute.MultiWorkerMirroredStrategy()

            if strategy:
                with strategy.scope():
                    run()
            else:
                run()
        else:
            with tf.Graph().as_default():
                trainer.set_random_seed(param.seed)
                with trainer.tf_device():
                    run(tf1_mode=True)


if __name__ == "__main__":
    _main()
