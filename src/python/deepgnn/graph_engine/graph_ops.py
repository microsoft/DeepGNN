# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Useful functions to work with graph that are not part of it's API."""
from typing import Tuple, Optional
import numpy as np
from deepgnn.graph_engine import Graph, FeatureType


def sample_out_edges(
    graph: Graph,
    nodes: np.ndarray,
    edge_types: np.ndarray,
    count: int,
    sampling_strategy: str = "byweight",
    edge_feature_meta: np.ndarray = None,
    edge_feature_type: FeatureType = FeatureType.FLOAT,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Get out edges for each nodes and edge features.

    e.g.: `src--(edge,type:1)-->dst`, return the out edge of `src` node is [src, dst, 1].

    Args:
      * graph: graph client.
      * nodes: src node.
      * edge_types: edge types.
      * count: how many edges will be returne for each src node.
      * sampling_strategy: samping strategy for node out edges: ["byweight", "topk", "random", "randomwithoutreplacement"]
      * edge_feature_meta: feature metadata.
        For example, `edge_feature_meta = np.array([[0, 2], [1, 3]], dtype=np.int32)`
        - feature index = 0, feature dimension = 2
        - feature index = 1, feature dimension = 3
      * edge_feature_type: feauture value type: [FeatureType.FLOAT|BINARY|INT64]

    Return:
      * edges: out edges of `nodes`, `edges.shape == (len(nodes) * count, 3)`.
      * features: edge features. return `None` if `edge_feature_meta is None`
    """
    nbrs, nbr_weights, nbr_types, nbr_cnt = graph.sample_neighbors(
        nodes, edge_types, count, sampling_strategy
    )
    assert nbrs.shape == (len(nodes), count)

    # edges shape: (#edge, 3)
    # - edges[:, 0]: src node.
    # - edges[:, 1]: dst node.
    # - edges[:, 2]: edge type.
    num_edges = len(nodes) * count
    edges = np.empty((num_edges, 3), dtype=np.int64)
    edges[:, 0] = np.repeat(nodes, count).reshape(-1)
    edges[:, 1] = nbrs.reshape(-1)
    edges[:, 2] = nbr_types.reshape(-1)

    feat = None
    if edge_feature_meta is not None:
        feat = graph.edge_features(edges, edge_feature_meta, edge_feature_type)
    return edges, feat


def get_skipgrams_size(path_len: int, left_win_size: int, right_win_size: int) -> int:
    """Compute skipgrams size."""
    pair_count = 0
    assert path_len > 0
    for i in range(path_len):
        pair_count += (i - max(0, i - left_win_size)) + (
            min(path_len - 1, i + right_win_size) - i
        )
    return pair_count


def gen_skipgrams(
    paths: np.ndarray, left_win_size: int, right_win_size: int
) -> np.ndarray:
    """Generate skipgram word pairs."""
    batch_size = paths.shape[0]
    path_len = paths.shape[1]
    pair_count = get_skipgrams_size(path_len, left_win_size, right_win_size)

    pairs = np.empty((batch_size * pair_count, 2), dtype=paths.dtype)
    idx = 0
    for i in range(0, batch_size):
        path = paths[i]
        # each path
        for j in range(0, path_len):
            left_start = max(0, j - left_win_size)
            right_end = min(path_len - 1, j + left_win_size)
            for k in range(left_start, right_end + 1):
                if k == j:
                    continue
                pairs[idx][0] = path[j]
                pairs[idx][1] = path[k]
                idx += 1
    return pairs


def sub_graph(
    graph: Graph,
    src_nodes: np.ndarray,
    edge_types: np.ndarray,
    num_hops: int = 1,
    self_loop: bool = True,
    undirected: bool = True,
    return_edges: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate n-hops neighbor graph for src nodes.

    Args:
      * graph: graph client.
      * src_nodes: seed nodes.
      * edge_types: edge types for neighbor graph.
      * num_hops: num of hops for neighbor graph.
      * self_loop: add self loop into adjacency matrix.
      * undirected: add reverse edges for undirected graph.
      * return_edges: False(return adjacency), True(return edges).

    Return:
      * nodes: all nodes in neighbor graph.
      * adj: adjacency matrix or edge list.
      * src_nodes_idx: the index of `src_nodes` in returned `nodes`. (nodes[src_nodes_idx] == src_nodes)
    """
    nodes = [src_nodes]
    for i in range(num_hops - 1):
        nb, _, _, _ = graph.neighbors(nodes[i], edge_types)
        tmp = np.unique(nb)
        nodes.append(tmp)

    nodes = np.concatenate(nodes)
    unodes = np.unique(nodes)
    nb, _, _, cnt = graph.neighbors(unodes, edge_types)

    # get all edges
    num_edges = np.sum(cnt)
    edges = np.zeros((num_edges, 2), np.int64)
    offset = 0
    for u, ucnt in zip(unodes, cnt):
        ucnt = int(ucnt)
        edges[offset : offset + ucnt, 0] = u
        offset += int(ucnt)
    edges[:, 1] = nb

    # use subgraph(edges) to build adj matrix.
    all_nodes = np.concatenate([src_nodes, edges.reshape(-1)])
    unique_nodes, idx = np.unique(all_nodes, return_inverse=True)
    src_nodes_idx = idx[: src_nodes.size]
    edge_idx = idx[src_nodes.size :]
    n = unique_nodes.size

    if return_edges:
        edge_idx = edge_idx.reshape(-1, 2)  # [num_edges, 2]
        if undirected:
            edge_idx_double = np.concatenate([edge_idx, edge_idx])
            # add reverse edge.
            edge_idx_double[edge_idx.shape[0] :, 0] = edge_idx[:, 1]
            edge_idx_double[edge_idx.shape[0] :, 1] = edge_idx[:, 0]
            edge_idx = edge_idx_double
        if self_loop:
            loop_edge = np.arange(n, dtype=np.int64)
            loop_edge = np.repeat(loop_edge, 2).reshape(-1, 2)
            edge_idx = np.concatenate([edge_idx, loop_edge])

        if self_loop or undirected:
            # remove duplicated edges
            edge_idx = np.unique(edge_idx, axis=0)

        return unique_nodes, edge_idx, src_nodes_idx

    else:
        adj = np.zeros((n, n), dtype=np.float32)
        edge_idx = edge_idx.reshape(-1, 2)
        adj[edge_idx[:, 0], edge_idx[:, 1]] = 1
        if undirected:
            adj[edge_idx[:, 1], edge_idx[:, 0]] = 1

        # add self loop
        if self_loop:
            adj += np.eye(n)

        return unique_nodes, adj, src_nodes_idx
