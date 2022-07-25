# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Conversion functions to internal binary format."""
import multiprocessing as mp
import typing
import platform

if platform.system() == "Windows":
    from multiprocessing.connection import PipeConnection as Connection  # type: ignore
else:
    from multiprocessing.connection import Connection  # type: ignore

import deepgnn.graph_engine.snark.converter.writers as writers
from deepgnn.graph_engine.snark.decoders import DecoderType


FLAG_ALL_DONE = b"WORK_FINISHED"
FLAG_WORKER_FINISHED_PROCESSING = b"WORKER_FINISHED_PROCESSING"
PROCESS_PRINT_INTERVAL = 1000


def converter_process(
    q_in: typing.Union[mp.Queue, Connection],
    q_out: mp.Queue,
    folder: str,
    suffix: int,
    decoder: DecoderType,
    skip_node_sampler: bool,
    skip_edge_sampler: bool,
) -> None:
    """Process graph nodes from a queue to binary files.

    Args:
        q_in (typing.Union[mp.Queue, Connection]): input nodes
        q_out (mp.Queue): signal processing is done
        folder (str): where to save binaries
        suffix (int): file suffix in the name of binary files
        decoder (Decoder): Decoder object which is used to parse the raw graph data file.
        skip_node_sampler(bool): skip generation of node alias tables
        skip_edge_sampler(bool): skip generation of edge alias tables
    """
    if isinstance(decoder, type):
        decoder = decoder()

    binary_writer = writers.BinaryWriter(
        str(folder),
        suffix,
        skip_node_sampler,
        skip_edge_sampler,
    )
    while True:
        if type(q_in) == Connection:
            line = q_in.recv()  # type: ignore
        else:
            line = q_in.get()  # type: ignore

        if line == FLAG_ALL_DONE:
            break

        binary_writer.add(decoder.decode(line))  # type: ignore

    binary_writer.close()
    q_out.put(
        (
            FLAG_WORKER_FINISHED_PROCESSING,
            {
                "node_count": binary_writer.node_count,
                "edge_count": binary_writer.edge_count,
                "node_type_num": binary_writer.node_type_num,
                "edge_type_num": binary_writer.edge_type_num,
                "node_feature_num": binary_writer.node_feature_num,
                "edge_feature_num": binary_writer.edge_feature_num,
                "partition": {
                    "id": suffix,
                    "node_weight": binary_writer.node_weight,
                    "node_type_count": binary_writer.node_type_count,
                    "edge_weight": binary_writer.edge_weight,
                    "edge_type_count": binary_writer.edge_type_count,
                },
            },
        )
    )
