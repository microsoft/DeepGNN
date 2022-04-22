# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import numpy as np
import pytest
import urllib.request
import tarfile
import tempfile
from collections import defaultdict
from typing import Tuple
from torch.utils.data import IterableDataset
from deepgnn.graph_engine import Graph, SamplingStrategy, FeatureType


@pytest.fixture(scope="module")
def prepare_local_test_files():
    name = "cora.tgz"
    working_dir = tempfile.TemporaryDirectory()
    zip_file = os.path.join(working_dir.name, name)
    urllib.request.urlretrieve(
        f"https://deepgraphpub.blob.core.windows.net/public/testdata/{name}", zip_file
    )
    with tarfile.open(zip_file) as tar:
        tar.extractall(working_dir.name)

    yield working_dir.name
    working_dir.cleanup()


def load_data(test_rootdir, num_feats=1433):
    num_nodes = 2708
    feat_data = np.zeros((num_nodes, num_feats))
    labels = np.empty((num_nodes, 1), dtype=np.int64)
    node_map = {}
    label_map = {}
    with open(os.path.join(test_rootdir, "cora", "cora.content")) as fp:
        for i, line in enumerate(fp):
            info = line.strip().split()
            feat_data[i, :] = [float(x) for x in info[1 : num_feats + 1]]
            node_map[info[0]] = i
            if not info[-1] in label_map:
                label_map[info[-1]] = len(label_map)
            labels[i] = label_map[info[-1]]

    adj_lists = defaultdict(list)
    with open(os.path.join(test_rootdir, "cora", "cora.cites")) as fp:
        for i, line in enumerate(fp):
            info = line.strip().split()
            paper1 = node_map[info[0]]
            paper2 = node_map[info[1]]
            adj_lists[paper1].append(paper2)
            adj_lists[paper2].append(paper1)

    return feat_data, labels, adj_lists


class MockSimpleDataLoader(IterableDataset):
    def __init__(self, batch_size: int, query_fn, graph: Graph):
        self.curr_node = 0
        self.batch_size = batch_size
        self.query_fn = query_fn
        self.graph = graph

    def __iter__(self):
        return self

    def __next__(self):
        res = self.curr_node
        self.curr_node += self.batch_size
        if self.curr_node >= 1025:
            raise StopIteration
        inputs = np.array([res + n for n in range(self.batch_size)]).astype(np.int64)
        context = self.query_fn(self.graph, inputs)
        return context


class MockFixedSimpleDataLoader(IterableDataset):
    def __init__(self, inputs: np.array, query_fn, graph: Graph):
        self.inputs = inputs
        self.query_fn = query_fn
        self.graph = graph

    def __iter__(self):
        return self

    def __next__(self):
        return self.query_fn(self.graph, self.inputs)


class MockGraph(Graph):
    def __init__(self, feat_data, labels, adj_lists):
        self.feat_data = feat_data
        self.labels = labels
        self.adj_lists = adj_lists
        self.type_ranges = [(0, 1000), (1000, 1500), (1500, len(adj_lists))]

    def sample_nodes(
        self,
        size: int,
        node_type: int,
        strategy: SamplingStrategy = SamplingStrategy.Random,
    ) -> np.array:
        return np.random.randint(
            self.type_ranges[node_type][0], self.type_ranges[node_type][1], size
        )

    def sample_neighbors(
        self,
        nodes: np.array,
        edge_types: np.array,
        count: int = 10,
        strategy: str = "byweight",
        default_node: int = -1,
        default_weight: float = 0.0,
        default_node_type: int = -1,
    ) -> Tuple[np.array, np.array, np.array, np.array]:
        res = np.empty((len(nodes), count), dtype=np.int64)
        nodes = nodes.reshape(-1)
        for i in range(len(nodes)):
            universe = np.array(self.adj_lists[nodes[i]], dtype=np.int64)

            # If there are no neighbors, fill results with a dummy value.
            if len(universe) == 0:
                res[i] = np.full(count, -1, dtype=np.int64)
            else:
                repetitions = int(count / len(universe)) + 1
                res[i] = np.resize(np.tile(universe, repetitions), count)

        return (
            res,
            np.full((len(nodes), count), 0.0, dtype=np.float32),
            np.full((len(nodes), count), -1, dtype=np.int32),
            np.full((len(nodes)), 0, dtype=np.int32),
        )

    def node_features(
        self, nodes: np.array, features: np.array, feature_type: FeatureType
    ) -> np.array:
        if np.array_equal(features, np.array([[1, 7]])):
            rt = []
            for n in nodes:
                lb = np.zeros(features[0][1])
                lb[self.labels[n]] = 1
                rt.append(lb)
            return np.array(rt, dtype=np.float32)

        return np.array([self.feat_data[n] for n in nodes], dtype=np.float32)


@pytest.fixture(scope="module")
def mock_graph(prepare_local_test_files):
    feat_data, labels, adj_lists = load_data(prepare_local_test_files)
    assert len(adj_lists) == 2708
    assert len(labels) == 2708
    assert len(feat_data) == 2708

    np.random.RandomState(seed=1)

    return MockGraph(feat_data, labels, adj_lists)
