# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Graph engine configuration."""
import argparse
from deepgnn.arg_types import str2bool
from deepgnn.graph_engine.backends.options import GraphType
from deepgnn.graph_engine.snark.converter.options import DataConverterType
from deepgnn.graph_engine.graph_dataset import BackendType


def define_param_graph_engine(parser: argparse.ArgumentParser):
    """Define command line arguments for graph engine."""
    group = parser.add_argument_group("Graph Client Parameters")
    group.add_argument(
        "--graph_type",
        type=GraphType,
        default=GraphType.LOCAL,
        choices=list(GraphType),
        help="create local/remote graph client.",
    )
    group.add_argument(
        "--data_dir",
        type=str,
        default="",
        help="graph data dir (local path or azure data lake path).",
    )

    # SSL
    group.add_argument(
        "--enable_ssl",
        type=str2bool,
        default=False,
        help="Enable TLS between graph client and server.",
    )
    group.add_argument(
        "--ssl_cert",
        type=str,
        default="",
        help="Server certificate path when enabling TLS.",
    )

    group.add_argument(
        "--backend",
        type=BackendType,
        choices=list(BackendType),
        default=BackendType.SNARK,
        help="Graph engine backend types.",
    )
    parser.add_argument("--servers", nargs="+", help="Snark server list", default="")
    parser.add_argument(
        "--converter",
        type=DataConverterType,
        default=DataConverterType.LOCAL,
        choices=list(DataConverterType),
        help="Converter types to convert raw graph data to binary.",
    )
    parser.add_argument(
        "--num_ge",
        type=int,
        default=0,
        help="Number of graph engine should be started, if 0, backend will calculate it.",
    )
    parser.add_argument(
        "--ge_start_timeout",
        type=int,
        required=False,
        default=30,
        help="Timeout in minutes when starting graph engine.",
    )
    parser.add_argument(
        "--GE_OMP_NUM_THREADS",
        type=int,
        default=1,
        help="GE openmp threads count (env variable: OMP_NUM_THREADS).",
    )
    group.add_argument(
        "--server_idx",
        type=int,
        default=None,
        help="Index of a GE instance from the servers argument to start",
    )
    group.add_argument(
        "--client_rank",
        type=int,
        default=None,
        help="A world rank for a GE client that can't be obtained from MPI/DDP environment variables",
    )
    group.add_argument(
        "--skip_ge_start",
        type=str2bool,
        default=False,
        help="Attach to exsiting GE servers instead of starting them with workers.",
    )
    group.add_argument(
        "--sync_dir",
        type=str,
        default="",
        help="Directory to synchronize start of graph engine and workers.",
    )
