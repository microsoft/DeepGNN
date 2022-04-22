# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Diagnostic tool built to obtain the properties of a NetworkX graph."""
import networkx as nx
import math
import random


def densification(g: nx.digraph) -> float:
    """Return densification constant in NetworkX directed graph."""
    return math.log(nx.number_of_edges(g), nx.number_of_nodes(g))


def diameter(g: nx.digraph) -> int:
    """Return effective 90% diameter in NetworkX directed graph."""
    largest_strongly_connected_component = g.subgraph(
        max(nx.strongly_connected_components(g), key=len)
    )
    return nx.diameter(largest_strongly_connected_component)


def largest_connected_component(g: nx.digraph) -> float:
    """Return the scaled size of the largest strongly connected component in NetworkX directed graph."""
    largest_strongly_connected_component = g.subgraph(
        max(nx.strongly_connected_components(g), key=len)
    )
    return nx.number_of_nodes(
        largest_strongly_connected_component
    ) / nx.number_of_nodes(g)


def max_adjacency(g: nx.digraph) -> float:
    """Return the largest eigenvalue of the adjacency matrix in NetworkX directed graph."""
    return max(nx.adjacency_spectrum(g))


def average_clustering(input_graph: nx.digraph, trials: int) -> float:
    """Return the average clustering coefficient in NetworkX directed graph."""
    g = nx.to_undirected(input_graph)
    n = len(g)
    triangles = 0
    nodes = g.nodes()
    for i in [random.randint(0, n) for _ in range(trials)]:
        nbrs = list(g[nodes[i]])
        if len(nbrs) < 2:
            continue
        u, v = random.sample(nbrs, 2)
        if u in g[v]:
            triangles += 1
    return triangles / float(trials)
