# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Multihop neighbor sampling on graph."""
import numpy as np
from typing import Tuple
from deepgnn.graph_engine._base import Graph


def sample_fanout(
    graph: Graph,
    nodes: np.ndarray,
    metapath: list,
    fanouts: list,
    default_node: int,
    sampling_strategy: str = "byweight",
) -> Tuple[list, list, list]:
    """
    Sample multi-hop neighbors of nodes according to weight in graph.

    Args:
    nodes: input nodes.
    metapath: Edge types to filter outgoing edges in each hop.
    fanouts: Number of sampling for each node in each hop.
    default_node: The node id to fill when there is no neighbor for specific nodes.

    Return:
    A tuple of lists: (samples, weights, types)
    samples: A list of `np.array`s of `int64`, with the same length as
        `edge_types` and `counts`, with shapes `[num_nodes]`,
        `[num_nodes * count1]`, `[num_nodes * count1 * count2]`, ...
    weights: A list of `np.array`s of `float32`, with shapes
        `[num_nodes * count1]`, `[num_nodes * count1 * count2]` ...
    types: A list of `np.array`s of `int32`, with shapes
        `[num_nodes * count1]`, `[num_nodes * count1 * count2]` ...
    """
    neighbors_list = [np.reshape(nodes, [-1])]
    weights_list = []
    types_list = []
    for hop_edge_types, count in zip(metapath, fanouts):
        res = graph.sample_neighbors(
            nodes=neighbors_list[-1],
            edge_types=np.array(hop_edge_types, dtype=np.int32),
            count=count,
            strategy=sampling_strategy,
            default_node=default_node,
        )
        neighbors_list.append(np.reshape(res[0], [-1]))
        weights_list.append(np.reshape(res[1], [-1]))
        types_list.append(np.reshape(res[2], [-1]))
    return neighbors_list, weights_list, types_list


def _full_neighbor(
    graph: Graph, nodes: np.ndarray, hop_edge_types: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(nodes) == 0:
        return (
            np.array([], dtype=np.int64),
            np.array([], dtype=np.float32),
            np.array([], dtype=np.int64).reshape(0, 2),
        )

    full_nbs = graph.neighbors(
        nodes, hop_edge_types
    )  # Tuple of (neighbors, weights, types, counts)
    blocks: list = []

    # Create neighbors indices in the form [node_idx, neighbor_idx]
    for x, i in np.nditer([full_nbs[3], np.arange(len(full_nbs[3]))]):
        blocks.append(
            np.block([np.full([x], i).reshape(-1, 1), np.arange(x).reshape(-1, 1)])  # type: ignore
        )
    return (
        full_nbs[0].astype(np.int64, copy=False),
        full_nbs[1],
        np.concatenate(blocks).astype(np.int64, copy=False),
    )


def get_neighbor(
    graph: Graph, nodes: np.ndarray, edge_types: np.ndarray, max_neighbors_per_node
) -> Tuple[list, list]:
    """
    Get multi-hop neighbors with adjacent matrix.

    Args:
    nodes: A 1-D `np.array` of `int64`.
    edge_types: Edge types to filter outgoing edges in each hop.

    Return:
    A tuple of list: (nodes, adjcents)
    nodes: A list of N + 1 `np.array` of `int64`, N is the number of hops.
    adjcents: A list of N tuples suitable to create `tf.SparseTensor`.
    """
    nodes = np.reshape(nodes, [-1])
    nodes_list = [[nodes, np.array([len(nodes)], dtype=np.int64)]]
    adj_list = []

    for hop_edge_types in edge_types:
        nodes_without_padding = nodes_list[-1][0][0 : nodes_list[-1][1][0]]
        neighbor_values, weight_values, indices = _full_neighbor(
            graph, nodes_without_padding, hop_edge_types
        )

        next_nodes, next_idx = np.unique(neighbor_values, return_inverse=True)
        next_indices = np.stack([indices[:, 0], next_idx], 1)
        next_values = weight_values
        next_shape = np.stack([len(nodes_without_padding), len(next_nodes)])
        actual_ind_length = len(next_indices)
        if actual_ind_length > max_neighbors_per_node * len(nodes_list[-1][0]):
            raise UserWarning(
                "Failed to pad sparse vector dimension {0} is larger than expected {1}".format(
                    actual_ind_length, max_neighbors_per_node * len(nodes_list[-1][0])
                )
            )

        next_adj = [
            np.pad(
                next_indices,
                (
                    (
                        0,
                        max_neighbors_per_node * len(nodes_list[-1][0])
                        - actual_ind_length,
                    ),
                    (0, 0),
                ),
            ),
            np.pad(
                next_values,
                (
                    0,
                    max_neighbors_per_node * len(nodes_list[-1][0]) - actual_ind_length,
                ),
            ),
            next_shape,
            np.array([actual_ind_length], dtype=np.int64),
        ]

        actual_node_length = len(next_nodes)
        next_nodes = np.pad(
            next_nodes,
            (0, len(nodes_list[-1][0]) * max_neighbors_per_node - actual_node_length),
        )
        nodes_list.append([next_nodes, np.array([actual_node_length], dtype=np.int64)])
        adj_list.append(next_adj)
        nodes = next_nodes
    return nodes_list, adj_list
