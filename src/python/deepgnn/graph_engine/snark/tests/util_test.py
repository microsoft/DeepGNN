# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Unit test utils."""
from typing import Optional
import socket
from contextlib import closing

from deepgnn.graph_engine.snark.decoders import JsonDecoder


def json_to_edge_list_feature(features):
    def get_f(f):
        if f is None:
            return "uint8,0"
        elif isinstance(f, str):
            return f"binary,1,{f}"
        elif isinstance(f, tuple):
            coords, values = f
            if values.size == 0:
                return f"{values.dtype.name},0"
            if values.size == 1 and coords.size > 1:
                coords = coords.reshape((1, -1))
            if len(coords.shape) == 1:
                coordinates_str = ",".join(map(str, coords))
            else:
                coordinates_str = ",".join((",".join(map(str, c)) for c in coords))

            return f"{values.dtype.name},{values.size}/{coords.shape[1] if len(coords.shape) > 1 else 0},{coordinates_str},{','.join(map(str, values))}"
        return f"{f.dtype.name},{f.size},{','.join(map(str, f))}"

    return ",".join(get_f(f) for f in features)


def edge_list_encode(
    node_id: int,
    blank: int,
    node_type: int,
    node_weight: float,
    node_created_at: int,
    node_deleted_at: int,
    node_features: list,
    edges: list,
    buffer: Optional[object] = None,  # type: ignore
    is_temporal: bool = False,
):
    """
    Convert data to a line of the edge_list format.

    Parameters
    ----------
    node_id: int
    node_type int
    node_weight: float
    node_features: [ndarray, ...]
    edges: [(edge_src: int, edge_dst: int, edge_type: int, edge_weight: float, edge_features[ndarray, ...]), ...]
    """
    node_contents = [node_id, -1, node_type, node_weight]
    if is_temporal:
        node_contents += [node_created_at, node_deleted_at]
    node_contents += [json_to_edge_list_feature(node_features)]
    buffer.write(",".join(map(str, node_contents)) + "\n")

    for (
        edge_src,
        edge_dst,
        edge_type,
        edge_weight,
        edge_created_at,
        edge_deleted_at,
        edge_features,
    ) in edges:
        min_creation = (
            node_created_at
            if edge_created_at == -1
            else max(node_created_at, edge_created_at)
        )
        min_deletion = (
            node_deleted_at
            if edge_deleted_at == -1
            else min(node_deleted_at, edge_deleted_at)
        )
        edge_contents = [edge_src, edge_type, edge_dst, edge_weight]
        if is_temporal:
            edge_contents += [min_creation, min_deletion]
        edge_contents += [json_to_edge_list_feature(edge_features)]
        buffer.write(",".join(map(str, edge_contents)) + "\n")


def json_node_to_edge_list(node, buffer, is_temporal=False):
    """Convert graph.json to graph.csv."""
    gen = JsonDecoder(is_temporal=is_temporal).decode(node)
    node = next(gen)
    edges = [edge for edge in gen]
    edge_list_encode(*node, edges, buffer, is_temporal)


def json_to_edge_list(filename_in, filename_out, is_temporal=False):
    """Convert graph.json to graph.csv."""
    file_in = open(filename_in, "r")
    file_out = open(filename_out, "w")
    while True:
        line = file_in.readline().strip()
        if not line:
            break
        json_node_to_edge_list(line, file_out, is_temporal=is_temporal)
    file_in.close()
    file_out.close()


def find_free_port():
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]
