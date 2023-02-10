# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Graph sampler for HetGnn model."""
import numpy as np

from deepgnn.graph_engine import Graph, SamplingStrategy


class HetGnnDataSampler:
    """
    Implementation of BaseSampler for HetGnn model.

    Training is based on heterogenous triple list.
    """

    def __init__(
        self,
        graph: Graph,
        num_nodes: int,
        batch_size: int,
        node_type_count: int,
        walk_length: int = 5,
        sample_files: str = "",
    ):
        """Initialize sampler."""
        self.graph = graph
        self.num_nodes = num_nodes
        self.nodes_left = num_nodes
        self.batch_size = batch_size
        self.node_type_count = node_type_count
        self.walk_length = walk_length
        self.count = int((self.num_nodes + self.batch_size - 1) / self.batch_size)
        self.samplers = []  # type: ignore

    # use N-hop random walk to generate positive/negtive samples.
    # once the dataset is initialized, all the triple list needed by this epoch is
    # created.
    def _generate_random_walk(self):
        triple_list = [[] for k in range(self.node_type_count**2)]

        triple_list_index = 0
        # Sample nodes based on the node type.
        for node_type_index in range(self.node_type_count):
            # get start node to do the random walk
            if len(self.samplers) > 0:
                c_nodes = next(iter(self.samplers[node_type_index]))
            else:
                c_nodes = self.graph.sample_nodes(
                    self.batch_size, node_type_index, strategy=SamplingStrategy.Weighted
                )

            # begin to random walk in N hops.
            neighbors_list = [c_nodes]
            neighbors_types_list = [np.full((len(c_nodes)), node_type_index)]
            for hop in range(self.walk_length):
                neighbors, _, neigh_types, _ = self.graph.sample_neighbors(
                    neighbors_list[-1],
                    np.array([k for k in range(self.node_type_count)], dtype=np.int32),
                    1,
                )
                neighbors_list.append(np.reshape(neighbors, (-1)))
                neighbors_types_list.append(np.reshape(neigh_types, (-1)))

            neighbors_list = np.reshape(neighbors_list, (self.walk_length + 1, -1)).T
            neighbors_types_list = np.reshape(
                neighbors_types_list, (self.walk_length + 1, -1)
            ).T

            for n in range(len(neighbors_list)):
                central_node = neighbors_list[n][0]
                for m in range(1, len(neighbors_list[n])):
                    if neighbors_types_list[n][m] == -1:
                        continue
                    triple_list_index = (
                        neighbors_types_list[n][0] * self.node_type_count
                        + neighbors_types_list[n][m]
                    )
                    triple_list[triple_list_index].append(
                        [central_node, neighbors_list[n][m], 0]
                    )

        for i in range(len(triple_list)):
            if len(triple_list[i]) > 0:
                neg_node = self.graph.sample_nodes(
                    len(triple_list[i]),
                    i % self.node_type_count,
                    strategy=SamplingStrategy.Weighted,
                )
                for k in range(len(triple_list[i])):
                    triple_list[i][k][2] = neg_node[k]

        return triple_list

    def __len__(self):
        """Return number of minibatches in this sampler."""
        return self.count

    def __iter__(self):
        """Implement IterableDataset method to provide data iterator."""
        self.nodes_left = self.num_nodes
        return self

    def __next__(self):
        """Implement iterator interface."""
        if self.nodes_left <= 0:
            raise StopIteration

        triple_list = self._generate_random_walk()
        self.nodes_left -= self.batch_size
        return triple_list
