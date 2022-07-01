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
from deepgnn.graph_engine.snark.decoders import DecoderType, JsonDecoder


FLAG_ALL_DONE = b"WORK_FINISHED"
FLAG_WORKER_FINISHED_PROCESSING = b"WORKER_FINISHED_PROCESSING"
PROCESS_PRINT_INTERVAL = 1000


class _NoOpWriter:
    def add(self, *_: typing.Any):
        return

    def close(self):
        return


def converter_process(
    q_in: typing.Union[mp.Queue, Connection],
    q_out: mp.Queue,
    folder: str,
    suffix: int,
    decoder: typing.Optional[DecoderType],
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
    if decoder is None:
        decoder = JsonDecoder()  # type: ignore

    node_count = 0
    edge_count = 0
    node_type_num = -1
    edge_type_num = -1

    node_feature_num = 5
    edge_feature_num = 5

    node_weight = []
    node_type_count = []
    edge_weight = []
    edge_type_count = []
    node_writer = writers.NodeWriter(str(folder), suffix)
    edge_writer = writers.EdgeWriter(str(folder), suffix)
    node_alias: typing.Union[writers.NodeAliasWriter, _NoOpWriter] = (
        _NoOpWriter()
        if skip_node_sampler
        else writers.NodeAliasWriter(str(folder), suffix)
    )
    edge_alias: typing.Union[writers.EdgeAliasWriter, _NoOpWriter] = (
        _NoOpWriter()
        if skip_edge_sampler
        else writers.EdgeAliasWriter(str(folder), suffix)
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

        for src, dst, typ, weight, features in decoder.decode(line):  # type: ignore
            if src == -1:
                node_writer.add(dst, typ, features)
                edge_writer.add_node()
                node_alias.add(dst, typ, weight)
                if typ > node_type_num:
                    for _ in range(typ - node_type_num):
                        node_alias.add_type()
                        node_weight.append(0)
                        node_type_count.append(0)
                    node_type_num = typ
                node_weight[typ] += float(weight)
                node_type_count[typ] += 1
                node_count += 1
            else:
                edge_writer.add(dst, typ, weight, features)
                if typ > edge_type_num:
                    for _ in range(typ - edge_type_num):
                        edge_alias.add_type()
                        edge_weight.append(0)
                        edge_type_count.append(0)
                    edge_type_num = typ
                edge_alias.add(src, dst, typ, weight)
                edge_weight[typ] += weight
                edge_type_count[typ] += 1
                edge_count += 1

    node_writer.close()
    edge_writer.close()
    node_alias.close()
    edge_alias.close()
    q_out.put(
        (
            FLAG_WORKER_FINISHED_PROCESSING,
            {
                "node_count": node_count,
                "edge_count": edge_count,
                "node_type_num": node_type_num+1,
                "edge_type_num": edge_type_num+1,
                "node_feature_num": node_feature_num,
                "edge_feature_num": edge_feature_num,
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
