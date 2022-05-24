# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Command line arguments to configure training."""

import argparse
from deepgnn.graph_engine import define_param_graph_engine


# fmt: off
def init_model_common_args(parser: argparse.ArgumentParser):
    """Model configuration."""
    group = parser.add_argument_group("Model Common Parameters")
    group.add_argument("--feature_type", default="float", type=str, choices=["float", "uint64", "binary"], help="Feature type.")
    group.add_argument("--feature_idx", default=-1, type=int, help="Feature index.")
    group.add_argument("--feature_dim", default=0, type=int, help="Feature dimension.")
    group.add_argument("--label_idx", default=1, type=int, help="Label index.")
    group.add_argument("--label_dim", default=0, type=int, help="Label dimension.")
    group.add_argument("--dim", default=256, type=int, help="Dimension of embedding.")
    group.add_argument("--num_negs", default=5, type=int, help="Number of negative samplings.")
    group.add_argument("--fanouts", default=[10, 10], nargs="+", type=int, help="Graphsage fanouts.")
    group.add_argument("--neighbor_count", type=int, default=10, help="Number of neighbors to sample of each node")
    group.add_argument("--meta_dir", type=str, default="", help="Local meta data dir.")
    group.add_argument("--featenc_config", default=None, type=str, help="Config file name of feature encoder.")
    group.add_argument("--model_args", type=str, default="", help="Other arguments for model with json format.")
    group.add_argument("--node_type", default=0, type=int, help="Node type to train/evaluate model.")


def init_dataset_args(parser: argparse.ArgumentParser):
    """Dataset configuration."""
    group = parser.add_argument_group("Dataset Parameters")
    group.add_argument("--partitions", type=int, nargs="+", default=[0])
    group.add_argument("--sample_file", type=str, default="", help="File which contains node id to calculate the embedding. It could be filename pattern.")
    group.add_argument("--batch_size", default=512, type=int, help="Mini-batch size.")
    group.add_argument("--prefetch_size", default=16, type=int, help="Number of queries to prefetch.")
    group.add_argument("--num_parallel", default=2, type=int, help="Number of graph queries to run in parallel.")
    group.add_argument("--data_parallel_num", default=2, type=int, help="How many subprocesses to use for data loading.")
    group.add_argument("--prefetch_factor", default=2, type=int, help="Number of samples loaded in advance by each worker.")
    group.add_argument("--max_id", type=int, help="Max node id.")
    group.add_argument("--strategy", type=str, default="RandomWithoutReplacement", help="GraphEngine Sampler: Weighted/Random/RandomWithoutReplacement.")


def init_optimizer_args(parser: argparse.ArgumentParser):
    """Optimizer configuration."""
    group = parser.add_argument_group("Optimizer Parameters")
    group.add_argument("--learning_rate", default=0.01, type=float, help="Learning rate.")


def init_other_args(parser: argparse.ArgumentParser):
    """Misc configuration."""
    group = parser.add_argument_group("Other Parameters")
    group.add_argument("--seed", type=int, default=None, help="Random seed for initialization")
# fmt :on


def init_common_args(parser: argparse.ArgumentParser):
    """Configure all components."""
    define_param_graph_engine(parser)
    init_model_common_args(parser)
    init_dataset_args(parser)
    init_optimizer_args(parser)
    init_other_args(parser)
