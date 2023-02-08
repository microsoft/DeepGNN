# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
import tensorflow as tf
import numpy as np

from deepgnn.graph_engine import (
    SamplingStrategy,
    GENodeSampler,
    RangeNodeSampler,
    BackendOptions,
    create_backend,
)

from deepgnn import str2list_int, str2list2_int, setup_default_logging_config
from contextlib import closing
from deepgnn.tf import common
from deepgnn.tf.common.dataset import create_tf_dataset
from deepgnn.tf.common.trainer_factory import get_trainer

from han import HAN, HANQuery, HANQueryParamemter  # type: ignore
from functools import partial


# fmt: off
def define_param_han(parser):
    parser.add_argument("--batch_size", type=int, default=16, help="mini-batch size")
    parser.add_argument("--epochs", type=int, default=1, help="num of epochs for training")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="learning rate")

    parser.add_argument("--training_node_types", type=str2list_int, default="0", help="Node types for training sampling.",)
    parser.add_argument("--evaluate_node_types", type=str2list_int, default="0", help="Node types for training sampling.",)

    parser.add_argument("--edge_types", type=str2list2_int, default="0;1", help="Edge types for training sampling.",)

    parser.add_argument("--feature_idx", type=int, default=-1, help="feature index.")
    parser.add_argument("--feature_dim", type=int, default=0, help="feature dim.")

    parser.add_argument("--fanouts", type=str2list_int, default="10", help="fanouts.")

    parser.add_argument("--num_nodes", type=int, default=-1, help="Maximum node id.")
    parser.add_argument("--loss_name", type=str, default="softmax", choices=["softmax", "sigmoid"])
    parser.add_argument("--softmax_gamma", type=float, default=1.0, help="gamma for softmax loss.")

    parser.add_argument("--label_idx", type=int, default=0, help="label index.")
    parser.add_argument("--label_dim", type=int, default=16, help="label dim.")

    parser.add_argument("--head_num", type=str2list_int, default="8,8", help="head number for each layer.",)
    parser.add_argument("--hidden_dim", type=str2list_int, default="8,8", help="hidden dimension for each layer.",)
# fmt: on


def build_model(param):
    label_idx, label_dim = param.label_idx, param.label_dim
    if param.mode == common.args.TrainMode.INFERENCE:
        # inference job doesn't need label info.
        label_idx, label_dim = -1, 1

    para = HANQueryParamemter(
        edge_types=param.edge_types,
        nb_num=param.fanouts,
        feature_idx=param.feature_idx,
        feature_dim=param.feature_dim,
        label_idx=label_idx,
        label_dim=label_dim,
        max_id=param.num_nodes,
    )

    query_obj = HANQuery(para)

    model = HAN(
        edge_types=param.edge_types,
        nb_num=param.fanouts,
        head_num=param.head_num,
        hidden_dim=param.hidden_dim,
        max_id=param.num_nodes,
        feature_idx=param.feature_idx,
        feature_dim=param.feature_dim,
        label_idx=label_idx,
        label_dim=label_dim,
        loss_name=param.loss_name,
    )

    return model, query_obj


def _main():
    # setup default logging component.
    setup_default_logging_config(enable_telemetry=True)

    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter, allow_abbrev=False
    )
    common.args.import_default_parameters(parser)
    define_param_han(parser)

    param = parser.parse_args()
    common.args.log_all_parameters(param)

    # Supported Trainer:
    # * PSTrainer (parameter server)
    # * HorovodTFTrainer

    # use one device strategy. change to other strategies based on your
    # scenarios.
    with tf.distribute.get_strategy().scope():
        trainer = get_trainer(param)
        model, query_obj = build_model(param)
        backend = create_backend(
            BackendOptions(param), is_leader=(trainer.task_index == 0)
        )

        with closing(backend):
            if param.mode == common.args.TrainMode.TRAIN:
                tf_dataset, _ = create_tf_dataset(
                    sampler_class=GENodeSampler,
                    query_fn=query_obj.query_trainning,
                    backend=backend,
                    node_types=np.array(param.training_node_types, dtype=np.int32),
                    batch_size=param.batch_size,
                    sample_num=param.num_nodes,
                    num_workers=trainer.worker_size,
                    worker_index=trainer.task_index,
                    strategy=SamplingStrategy.Random,
                )

                trainer.train(
                    dataset=tf_dataset,
                    model=model,
                    optimizer=tf.compat.v1.train.AdamOptimizer(
                        param.learning_rate * trainer.lr_scaler
                    ),
                    epochs=param.epochs,
                )
            elif param.mode == common.args.TrainMode.EVALUATE:
                tf_dataset, _ = create_tf_dataset(
                    sampler_class=GENodeSampler,
                    query_fn=query_obj.query_trainning,
                    backend=backend,
                    node_types=np.array(param.evaluate_node_types, dtype=np.int32),
                    batch_size=param.batch_size,
                    sample_num=param.num_nodes,
                    num_workers=trainer.worker_size,
                    worker_index=trainer.task_index,
                    strategy=SamplingStrategy.Random,
                )

                trainer.evaluate(dataset=tf_dataset, model=model)
            elif param.mode == common.args.TrainMode.INFERENCE:
                invalid_id = param.num_nodes + 1
                tf_dataset, _ = create_tf_dataset(
                    sampler_class=RangeNodeSampler,
                    query_fn=query_obj.query_trainning,
                    backend=backend,
                    first=0,
                    last=param.num_nodes,
                    batch_size=param.batch_size,
                    num_workers=trainer.worker_size,
                    worker_index=trainer.task_index,
                    backfill_id=invalid_id,
                )

                valid_node_embedding_to_str = partial(
                    common.utils.node_embedding_to_string, invalid_id=invalid_id
                )
                trainer.inference(
                    dataset=tf_dataset,
                    model=model,
                    embedding_to_str_fn=valid_node_embedding_to_str,
                )
            else:
                raise NotImplementedError("not implement yet.")


if __name__ == "__main__":
    _main()
