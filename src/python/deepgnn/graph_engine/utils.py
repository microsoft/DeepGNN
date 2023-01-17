# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Graph engine configuration."""
from typing import List
import argparse
import numpy as np
from deepgnn.arg_types import str2bool
from deepgnn.graph_engine.snark.converter.options import DataConverterType


def define_param_graph_engine(parser: argparse.ArgumentParser):
    """Define command line arguments for graph engine."""
    group = parser.add_argument_group("Graph Client Parameters")
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


def serialize(inputs: List[np.ndarray], batch_size: int):
    """Serialize query output."""
    vector_sizes = [i.size for i in inputs]
    vector_shape_lens = [len(i.shape) for i in inputs]
    output_size_unbuffered = (
        1 + len(vector_shape_lens) + sum(vector_shape_lens) + sum(vector_sizes)
    )
    mult = int(np.ceil(output_size_unbuffered / batch_size))
    output = np.zeros(batch_size * mult, dtype=np.float64)
    output[0] = len(vector_sizes)
    i = 1
    for inpt in inputs:
        size = inpt.size
        shape = inpt.shape
        output[i] = len(shape)
        output[i + 1 : i + 1 + len(shape)] = shape
        i += 1 + len(shape)
        output[i : i + size] = inpt.flatten()
        i += size
    return output.reshape(batch_size, -1)


def deserialize(value: np.ndarray):
    """Deserialize query output."""
    value = value.flatten()
    output = []
    n_items = int(value[0])
    i = 1
    for item in range(n_items):
        shape_len = int(value[i])
        shape = [int(v) for v in value[i + 1 : i + 1 + shape_len]]
        size = int(np.product(shape))
        i += 1 + shape_len
        output.append(value[i : i + size].reshape(shape))
        i += size
    return output
