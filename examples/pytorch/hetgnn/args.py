# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse


# fmt: off
def init_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group("HetGnn Parameters")
    group.add_argument("--walk_length", default=5, type=int)
    group.add_argument("--node_type_count", type=int, default=2, help="number of node type in the graph")
# fmt: on
