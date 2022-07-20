# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Conversion functions to internal binary format."""
import multiprocessing as mp
import typing
import platform
from typing import Optional

if platform.system() == "Windows":
    from multiprocessing.connection import PipeConnection as Connection  # type: ignore
else:
    from multiprocessing.connection import Connection  # type: ignore

import deepgnn.graph_engine.snark.converter.writers as writers
from deepgnn.graph_engine.snark.decoders import (
    DecoderType,
    Decoder,
    JsonDecoder,
    TsvDecoder,
)


FLAG_ALL_DONE = b"WORK_FINISHED"
FLAG_WORKER_FINISHED_PROCESSING = b"WORKER_FINISHED_PROCESSING"
PROCESS_PRINT_INTERVAL = 1000


class _NoOpWriter:
    def add(self, _: typing.Any):
        return

    def close(self):
        return


def converter_process(
    q_in: typing.Union[mp.Queue, Connection],
    q_out: mp.Queue,
    folder: str,
    suffix: int,
    node_type_num: int,
    edge_type_num: int,
    decoder_type: DecoderType,
    skip_node_sampler: bool,
    skip_edge_sampler: bool,
) -> None:
    """Process graph nodes from a queue to binary files.

    Args:
        q_in (typing.Union[mp.Queue, Connection]): input nodes
        q_out (mp.Queue): signal processing is done
        folder (str): where to save binaries
        suffix (int): file suffix in the name of binary files
        node_type_num (int): number of node types in the graph
        edge_type_num (int): number of edge types in the graph
        decoder_type (DecoderType): Decoder which is used to parse the raw graph data file, Supported: json/tsv
        skip_node_sampler(bool): skip generation of node alias tables
        skip_edge_sampler(bool): skip generation of edge alias tables
    """
    decoder: Optional[Decoder] = None
    if decoder_type == DecoderType.JSON:
        decoder = JsonDecoder()
    elif decoder_type == DecoderType.TSV:
        decoder = TsvDecoder()
    else:
        raise ValueError("Unsupported decoder type.")

    assert decoder is not None

    node_count = 0
    edge_count = 0
    node_weight = [0] * node_type_num
    node_type_count = [0] * node_type_num
    edge_weight = [0] * edge_type_num
    edge_type_count = [0] * edge_type_num
    writer = writers.NodeWriter(str(folder), suffix)
    node_alias: typing.Union[writers.NodeAliasWriter, _NoOpWriter] = (
        _NoOpWriter()
        if skip_node_sampler
        else writers.NodeAliasWriter(str(folder), suffix, node_type_num)
    )
    edge_alias: typing.Union[writers.EdgeAliasWriter, _NoOpWriter] = (
        _NoOpWriter()
        if skip_edge_sampler
        else writers.EdgeAliasWriter(str(folder), suffix, edge_type_num)
    )
    count = 0
    while True:
        count += 1
        if type(q_in) == Connection:
            line = q_in.recv()  # type: ignore
        else:
            line = q_in.get()  # type: ignore

        if line == FLAG_ALL_DONE:
            break

        node = decoder.decode(line)
        writer.add(node)
        node_alias.add(node)

        for eet in node["edge"]:
            edge_alias.add(eet)
            edge_weight[eet["edge_type"]] += eet["weight"]
            edge_type_count[eet["edge_type"]] += 1

        edge_count += len(node["edge"])
        node_count += 1
        node_weight[node["node_type"]] += float(node["node_weight"])
        node_type_count[node["node_type"]] += 1

    writer.close()
    node_alias.close()
    edge_alias.close()
    q_out.put(
        (
            FLAG_WORKER_FINISHED_PROCESSING,
            {
                "node_count": node_count,
                "edge_count": edge_count,
                "partition": {
                    "id": suffix,
                    "node_weight": node_weight,
                    "node_type_count": node_type_count,
                    "edge_weight": edge_weight,
                    "edge_type_count": edge_type_count,
                },
            },
        )
    )
