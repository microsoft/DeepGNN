# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
import tensorflow as tf
import numpy as np

from deepgnn.graph_engine import (
    SamplingStrategy,
    GENodeSampler,
    FileNodeSampler,
    BackendOptions,
    create_backend,
)

from deepgnn import str2list_int, str2bool, setup_default_logging_config
from contextlib import closing
from deepgnn.tf import common
from deepgnn.tf.common.dataset import create_tf_dataset, get_distributed_dataset
from deepgnn.tf.common.trainer_factory import get_trainer

from sage import SAGEQueryParameter, LayerInfo  # type: ignore
from sage_unsupervised import (  # type: ignore
    UnsupervisedGraphSAGE,
    UnsupervisedQuery,
    UnsupervisedSamplingParam,
)


# fmt: off
def define_param_graphsage(parser):
    parser.add_argument("--batch_size", type=int, default=512, help="mini-batch size")
    parser.add_argument("--epochs", type=int, default=1, help="num of epochs for training")
    parser.add_argument("--learning_rate", type=float, default=0.00001, help="learning rate")

    # GraphSAGE Model Parameters.
    parser.add_argument("--agg_type", type=str, default="mean", choices=["mean", "maxpool", "lstm"], help="aggregate functions.")
    parser.add_argument("--layer_dims", type=str2list_int, default="128,128", help="the output dimention for each layer (if concat=True, final dimension is 2x).")
    parser.add_argument("--num_classes", type=int, default=121, help="number of classes for category")
    parser.add_argument("--dropout", type=float, default=0.0, help="dropout rate.")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="weight for l2 loss on embedding matrix.")
    parser.add_argument("--loss_name", type=str, default="xent", choices=["xent"], help="loss function")
    parser.add_argument("--negative_sample_weight", type=float, default=1.0, help="weight for negative sample in loss function.")
    parser.add_argument("--identity_dim", type=int, default=-1, help="Set to positive value to use identity embedding features of that dimension.",)
    parser.add_argument("--all_node_count", type=int, default=0, help="all node count in graph.")

    # Node Sampler Setting:
    parser.add_argument("--node_files", type=str, default=None, help="if not None, use FileNodeSampler.")
    parser.add_argument("--node_types", type=str2list_int, default="0", help="if `node_files` is None, use GENodeSampler.")

    def register_query_param(parser):
        group = parser.add_argument_group("GraphSAGE Query Parameters")
        group.add_argument("--num_samples", type=str2list_int, default="25,10", help="number of samplers in each layer.",)
        group.add_argument("--neighbor_edge_types", type=str2list_int, default="0", help="edge types for neighbor sampling.",)
        group.add_argument("--feature_idx", type=int, default=-1, help="feature index.")
        group.add_argument("--feature_dim", type=int, default=-1, help="feature dim.")
        group.add_argument("--label_idx", type=int, default=-1, help="label index.")
        group.add_argument("--label_dim", type=int, default=-1, help="label dim.")
        group.add_argument("--negative_node_types", type=str2list_int, default="0", help="node types for negative sampling.")
        group.add_argument("--negative_num", type=int, default=20, help="num of nodes for negative sampling.")
        group.add_argument("--identity_feature", type=str2bool, default="False", help="whether use node feature.")

    register_query_param(parser)
# fmt: on


def build_model(param):
    nb_edges = np.array(param.neighbor_edge_types, np.int32)
    layer_infos = []
    assert len(param.layer_dims) == len(
        param.num_samples
    ), f"`layer_dims`({param.layer_dims}) doesn't match `num_samples`({param.num_samples})"
    for i in range(len(param.num_samples)):
        layer_infos.append(LayerInfo(param.num_samples[i], nb_edges, "random"))

    p = SAGEQueryParameter(
        layer_infos=layer_infos,
        feature_idx=param.feature_idx,
        feature_dim=param.feature_dim,
        label_idx=param.label_idx,
        label_dim=param.label_dim,
        identity_feature=param.identity_feature,
    )
    unsup_param = UnsupervisedSamplingParam(
        positive_edge_types=np.array(param.neighbor_edge_types, np.int32),
        positive_sampling_strategy="random",
        negative_node_types=np.array(param.negative_node_types, np.int32),
        negative_num=param.negative_num,
    )
    # fmt: off
    if param.identity_feature:
        num_nodes = param.all_node_count
        identity_embed_shape = [num_nodes + 1, param.identity_dim]
        assert (num_nodes > 0 and param.identity_dim > 0), f"use identity features (num_nodes * dim): {identity_embed_shape}"
        in_dim = param.identity_dim
        p.feature_idx = -1
        p.feature_dim = -1
    else:
        identity_embed_shape = None
        in_dim = param.feature_dim
    # fmt: on

    query_obj = UnsupervisedQuery(p, unsup_param)

    model = UnsupervisedGraphSAGE(
        in_dim=in_dim,
        layer_dims=param.layer_dims,
        num_samples=param.num_samples,
        dropout=param.dropout,
        loss_name=param.loss_name,
        agg_type=param.agg_type,
        weight_decay=param.weight_decay,
        negative_sample_weight=param.negative_sample_weight,
        identity_embed_shape=identity_embed_shape,
    )

    return model, query_obj


def run_train(param, trainer, query, model, tf1_mode, backend):
    if param.node_files is not None:
        tf_dataset, steps_per_epoch = create_tf_dataset(
            sampler_class=FileNodeSampler,
            query_fn=query.query_training,
            backend=backend,
            sample_files=param.node_files,
            batch_size=param.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=trainer.worker_size,
            worker_index=trainer.task_index,
        )

    else:
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
        lambda ctx: tf_dataset if tf1_mode else tf_dataset.repeat(param.epochs)
    )

    # we need to make sure the steps_per_epoch are provided in distributed dataset.
    assert steps_per_epoch is not None or param.steps_per_epoch is not None

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
        epochs=param.epochs,
        steps_per_epoch=(steps_per_epoch or param.steps_per_epoch),
    )


def run_eval(param, trainer, query, model, backend):
    tf_dataset, steps_per_epoch = create_tf_dataset(
        sampler_class=FileNodeSampler,
        query_fn=query.query_training,
        backend=backend,
        sample_files=param.node_files,
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
        sampler_class=FileNodeSampler,
        query_fn=query.query_training,
        backend=backend,
        sample_files=param.node_files,
        batch_size=param.batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=trainer.worker_size,
        worker_index=trainer.task_index,
    )

    distributed_dataset = get_distributed_dataset(lambda ctx: tf_dataset)

    # we need to make sure the steps_per_epoch are provided in distributed dataset.
    assert steps_per_epoch is not None or param.steps_per_epoch is not None

    trainer.inference(
        dataset=distributed_dataset,
        model=model,
        embedding_to_str_fn=common.utils.node_embedding_to_string,
        steps=(steps_per_epoch or param.steps_per_epoch),
    )


def _main():
    # setup default logging component.
    setup_default_logging_config(enable_telemetry=True)

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter, allow_abbrev=False
    )
    common.args.import_default_parameters(parser)
    define_param_graphsage(parser)

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
            # use mirrored strategy. change to other strategies based on your
            # scenarios.
            with tf.distribute.MirroredStrategy().scope():
                run()
        else:
            with tf.Graph().as_default():
                trainer.set_random_seed(param.seed)
                with trainer.tf_device():
                    run(tf1_mode=True)


if __name__ == "__main__":
    _main()
