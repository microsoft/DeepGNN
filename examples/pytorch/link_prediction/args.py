# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Command line arguments for link prediction model."""

import argparse
from deepgnn.arg_types import str2list_int, str2list2, str2list2_int


# fmt: off
def init_args(parser: argparse.ArgumentParser):
    """Initialize all arguments."""
    group = parser.add_argument_group("LinkPrediction Parameters")
    group.add_argument("--share_encoder", action="store_true", help="Whether or not to share the feature encoder for different node types.")

    # similarity config
    group.add_argument("--sim_type", type=str, default="cosine", help="Pooler type used in the crossing layer.")
    group.add_argument("--nsp_gamma", type=float, default=1, help="Negative sampling factor used in cosine with rng pooler.")
    group.add_argument("--weight_decay", type=float, default=0.01)
    group.add_argument("--use_mse", action="store_true", help="If MSELoss will be used.")
    group.add_argument("--res_size", type=int, default=64, help="Residual layer dimension.")
    group.add_argument("--res_bn", action="store_true", help="If BatchNorm1d is used in residual layers.",)
    group.add_argument("--label_map", type=str, default=None, help="This is used to normalize the labels in each task.")

    # gnn algo
    group.add_argument("--gnn_encoder", type=str, default="gat", help="Encoder name of GNN layer.")
    group.add_argument("--gnn_acts", type=str, default="leaky_relu,tanh", help="Activation functions used in GNN layer.")
    group.add_argument("--gnn_head_nums", type=str2list_int, default="2", help="Number of the heads.")
    group.add_argument("--gnn_hidden_dims", type=str2list_int, default="128", help="Hidden layer dimensions.")
    group.add_argument("--lgcl_largest_k", type=int, default=0, help="top k neighbors when using LGCL.")
    group.add_argument("--gnn_residual", type=str, default="add", help="residual layer type.")
    group.add_argument("--src_encoders", type=str2list2, default="q;k;s", help="Types of encoders used to encode feature of src nodes and their 1st/2nd hop neighbors.")
    group.add_argument("--dst_encoders", type=str2list2, default="k;q;s", help="Types of encoders used to encode feature of dst nodes and their 1st/2nd hop neighbors.")
    group.add_argument("--neighbor_mask", action="store_true", help="Nodes which need to be removed if masked.")

    # graph engine sampler
    group.add_argument("--src_metapath", type=str2list2_int, default="0;2", help="neighbor node types of source node which need to be sampled.")
    group.add_argument("--src_fanouts", type=str2list2_int, default="3;2", help="how many neighbors of source node will be sampled for each hop.")
    group.add_argument("--dst_metapath", type=str2list2_int, default="1;4", help="neighbor node types of destination node which need to be sampled.")
    group.add_argument("--dst_fanouts", type=str2list2_int, default="3;2", help="how many neighbors of destination node will be sampled for each hop.")
    group.add_argument("--train_file_dir", type=str, default="", help="Train file directory. It can be local path or adl path.")
    group.add_argument("--momentum", default=0.5, type=float, help="Used in optimizer.")

    # multi-task learning
    group.add_argument("--multi_task_config", type=str, default="", help="Multi-task learning config file name.")
# fmt: on
