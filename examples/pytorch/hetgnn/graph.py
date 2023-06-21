# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import csv
import numpy as np
from typing import Tuple

from torch.utils.data import IterableDataset
from deepgnn.graph_engine import Graph

_node_base_index = 1000000


class MockGraph(Graph):
    """Pure python implementation of Graph interface.

    Node ids in the graph are split in 3 types:
    * A nodes have ids [0.._node_base_index]
    * P nodes have ids [_node_base_index..2*_node_base_index]
    * V node are inside [2*_node_base_index+1..3*_node_base_index]

    Feature data stored in features list split by type above.
    Neighbors are stored as a single list split by node types above.
    We use nb_start_end field to find interval with nodes neighbors.
    """

    def __init__(self, feat_data, adj_lists, config):
        self.type_ranges = [
            (_node_base_index, _node_base_index + 2000),
            (_node_base_index * 2, _node_base_index * 2 + 2000),
            (_node_base_index * 3, _node_base_index * 3 + 10),
        ]
        self.features = [
            np.empty((config["A_n"], 128), dtype=np.float32),
            np.empty((config["P_n"], 128), dtype=np.float32),
            np.empty((config["V_n"], 128), dtype=np.float32),
        ]
        for i in range(len(feat_data)):
            for k, v in feat_data[i].items():
                self.features[i][int(k)] = v
        nb_counts = [sum(map(len, adj)) for adj in adj_lists]
        self.neighbors = [np.empty(count, dtype=np.int64) for count in nb_counts]
        self.nb_start_end = [
            np.empty((config["A_n"], 2), dtype=np.int32),
            np.empty((config["P_n"], 2), dtype=np.int32),
            np.empty((config["V_n"], 2), dtype=np.int32),
        ]
        for i in range(len(adj_lists)):
            start = 0
            for nb in range(len(adj_lists[i])):
                nb_count = len(adj_lists[i][nb])
                self.neighbors[i][start : start + nb_count] = np.array(
                    [self._map_node_id(x[1:], x[0]) for x in adj_lists[i][nb]],
                    dtype=np.int64,
                )
                self.nb_start_end[i][nb] = [start, start + nb_count]
                start += nb_count

    def sample_nodes(
        self,
        size: int,
        node_type: int,
        *args,
        **kwargs,
    ) -> np.ndarray:
        return np.random.randint(
            self.type_ranges[node_type][0], self.type_ranges[node_type][1], size
        )

    def _map_node_id(self, node_id, node_type):
        if node_type == "a" or node_type == "0":
            return int(node_id) + _node_base_index
        if node_type == "p" or node_type == "1":
            return int(node_id) + (_node_base_index * 2)
        if node_type == "v" or node_type == "2":
            return int(node_id) + (_node_base_index * 3)

    def sample_neighbors(
        self,
        nodes: np.ndarray,
        edge_types: np.ndarray,
        count: int = 10,
        *args,
        **kwargs,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        res = np.empty((len(nodes), count), dtype=np.int64)
        res_types = np.full((len(nodes), count), -1, dtype=np.int32)
        res_count = np.empty((len(nodes)), dtype=np.int64)
        for i in range(len(nodes)):
            if nodes[i] == -1:
                res[i] = np.full(count, -1, dtype=np.int64)
                res_count[i] = 0
                continue
            node_type = (nodes[i] // _node_base_index) - 1
            node_id = nodes[i] % _node_base_index
            interval = self.nb_start_end[node_type][node_id]

            # If there are no neighbors, fill results with a dummy value.
            if interval[0] == interval[1]:
                res[i] = np.full(count, -1, dtype=np.int64)
                res_count[i] = 0
                continue

            res[i] = np.random.choice(
                self.neighbors[node_type][interval[0] : interval[1]],
                count,
                replace=True,
            )
            res_count[i] = count

            for nt in range(len(res[i])):
                if res[i][nt] != -1:
                    neightype = (res[i][nt] // _node_base_index) - 1
                    res_types[i][nt] = neightype

        return (
            res,
            np.empty((len(nodes), count), dtype=np.float32),
            res_types,
            res_count,
        )

    def node_features(
        self,
        nodes: np.ndarray,
        features: np.ndarray,
        *args,
        **kwargs,
    ) -> np.ndarray:
        node_features = np.empty((len(nodes), features[0][1]), dtype=np.float32)
        for i in range(len(nodes)):
            node_id = nodes[i]
            node_type = (node_id // _node_base_index) - 1
            key = node_id % _node_base_index
            if node_id == -1 or key > len(self.features[node_type]):
                node_features[i] = np.zeros(features[0][1], dtype=np.float32)
            else:
                node_features[i] = self.features[node_type][key]
        return node_features

    def edge_count(self, *args, **kwargs):
        raise "Not needed for HetGNN"

    def edge_features(self, *args, **kwargs):
        raise "Not needed for HetGNN"

    def neighbors(self, *args, **kwargs):
        raise "Not needed for HetGNN"

    def node_count(self, *args, **kwargs):
        raise "Not needed for HetGNN"

    def node_types(self, *args, **kwargs):
        raise "Not needed for HetGNN"

    def random_walk(self, *args, **kwargs):
        raise "Not needed for HetGNN"

    def sample_edges(self, *args, **kwargs):
        raise "Not needed for HetGNN"


class MockHetGnnFileNodeLoader(IterableDataset):
    def __init__(
        self, graph: Graph, batch_size: int = 200, sample_file: str = "", model=None
    ):
        self.graph = graph
        self.batch_size = batch_size
        node_list = []
        with open(sample_file, "r") as f:
            data_file = csv.reader(f)
            for i, d in enumerate(data_file):
                node_list.append([int(d[0]) + _node_base_index, int(d[1])])
        self.node_list = np.array(node_list)
        self.cur_batch = 0
        self.model = model

    def __iter__(self):
        """Implement IterableDataset method to provide data iterator."""
        return self

    def __next__(self):
        """Implement iterator interface."""
        if self.cur_batch * self.batch_size >= len(self.node_list):
            raise StopIteration
        start_pos = self.cur_batch * self.batch_size
        self.cur_batch += 1
        end_pos = self.cur_batch * self.batch_size
        if end_pos >= len(self.node_list):
            end_pos = -1
        context = {}
        context["inputs"] = np.array(self.node_list[start_pos:end_pos][:, 0])
        context["node_type"] = self.node_list[start_pos:end_pos][0][1]
        context["encoder"] = self.model.build_node_context(
            context["inputs"], self.graph
        )
        return context
