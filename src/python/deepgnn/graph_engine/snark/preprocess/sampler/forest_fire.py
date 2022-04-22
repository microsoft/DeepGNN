# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""ForestFire samples a NetworkX graph through choosing random nodes and recursively burning out sections of the graph to include in the sample."""
import random
import networkx as nx
import numpy as np


def forest_fire(g: nx.digraph, sample_size: int) -> nx.digraph:
    """
    forest_fire takes in a NetworkX graph and a sample sample_size and returns a sample of the input graph of the input sample_size.

    Input Variables:
    g: NetworkX directed input graph.
    sample_size: Number of nodes to be in the returned sample graph.
    """
    assert nx.is_directed(g), "Input graph must be a directed graph"
    return __forest_fire_helper(g, sample_size, 0.37, 0.33)


def __forest_fire_helper(
    g: nx.digraph, sample_size: int, forward_prob: float, back_prob: float
) -> nx.digraph:
    """
    forest_fire_helper is here to test different forward and backward burning probabilities.

    Input Variables:
    g: NetworkX directed graph.
    sample_size: Number of nodes to be in the returned sample graph.
    forward_prob: probability that a forwards edge will be burned during forest fire.
    back_prob: probability that a backwards edge will be burned during forest fire.
    """
    sample = nx.DiGraph()

    list_nodes = list(g.nodes())

    visited = set()
    rand_node = list_nodes[random.randint(0, len(list_nodes) - 1)]

    q = []
    q.append(rand_node)
    while len(sample.nodes()) < sample_size:
        if len(q) > 0:
            initial_node = q.pop()
            # can revisit as long as not burned from same node
            if initial_node not in visited:
                visited.add(initial_node)
                successor = list(g.successors(initial_node))
                predecessor = list(g.predecessors(initial_node))
                num_successor = np.random.geometric(1 - forward_prob)
                num_predecessor = np.random.geometric(1 - forward_prob * back_prob)

                if num_successor > len(successor):
                    num_successor = len(successor)
                if num_predecessor > len(predecessor):
                    num_predecessor = len(predecessor)

                chosen_successor = random.sample(successor, num_successor)
                chosen_predecessor = random.sample(predecessor, num_predecessor)

                for succ in chosen_successor:
                    if len(sample.nodes()) < sample_size:
                        sample.add_edge(initial_node, succ)
                        q.append(succ)
                    else:
                        break

                for pred in chosen_predecessor:
                    if len(sample.nodes()) < sample_size:
                        sample.add_edge(pred, initial_node)
                        q.append(pred)
                    else:
                        break
        else:
            rand_node = list_nodes[(random.randint(0, len(list_nodes) - 1))]
            q.append(rand_node)
            visited.clear()
    return sample
