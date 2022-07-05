# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import pytest
from typing import Tuple, Union, Any
import numpy as np
import numpy.testing as npt
from deepgnn.graph_engine._base import Graph
from deepgnn.graph_engine import multihop


class NeighborGraph(Graph):
    def neighbors(
        self, nodes: np.ndarray, edge_types: Union[int, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return (
            np.array([[1, 2, 3]]),
            np.array([[1.0, 1.0, 1.0]]),
            np.array([[1, 1, 1]]),
            np.array([3]),
        )


@pytest.fixture(scope="module")
def neighbor_graph():
    return NeighborGraph()


def test_get_neighbor(neighbor_graph):
    res = multihop.get_neighbor(
        neighbor_graph, np.array([1]), np.array([1]), max_neighbors_per_node=6
    )
    assert len(res) == 2

    # initial nodes
    npt.assert_equal(res[0][0][0], np.array([1]))

    # length of initial nodes
    npt.assert_equal(res[0][0][1], np.array([1]))

    # 1st hop neighbors
    npt.assert_equal(res[0][1][0], np.array([1, 2, 3, 0, 0, 0]))

    # length of 1 hop neighbors
    npt.assert_equal(res[0][1][1], np.array([3]))

    # coordinates
    npt.assert_equal(
        res[1][0][0], np.array([[0, 0], [0, 1], [0, 2], [0, 0], [0, 0], [0, 0]])
    )
    # length of 1 hop neighbors
    npt.assert_equal(res[1][0][1][0], np.array([1.0, 1.0, 1.0, 0, 0, 0]))

    # shape
    npt.assert_equal(res[1][0][2], np.array([1, 3]))

    # length without padding
    npt.assert_equal(res[1][0][3], np.array([3]))


def test_get_neighbor_overfit(neighbor_graph):
    with pytest.raises(UserWarning):
        multihop.get_neighbor(
            neighbor_graph, np.array([1]), np.array([1]), max_neighbors_per_node=2
        )


class FanoutGraph(Graph):
    def sample_neighbors(
        self,
        nodes: np.ndarray,
        edge_types: Union[int, np.ndarray],
        count: int = 10,
        strategy: str = "byweight",
        default_node: int = -1,
        default_weight: float = 0.0,
        default_node_type: int = -1,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        adj_lists = [[1, 2, 3], [2, 3], [3], []]
        res = np.empty((len(nodes), count), dtype=np.int64)
        for i in range(len(nodes)):
            universe = np.array(adj_lists[nodes[i]], dtype=np.int64)

            # If there are no neighbors, fill results with a dummy value.
            if len(universe) == 0:
                res[i] = np.full(count, -1, dtype=np.int64)
            else:
                repetitions = int(count / len(universe)) + 1
                res[i] = np.resize(np.tile(universe, repetitions), count)

        return (
            res,
            np.full((len(nodes), count), 0.0, dtype=np.float32),
            np.full((len(nodes), count), 1, dtype=np.int32),
            np.full((len(nodes)), 0, dtype=np.int32),
        )


@pytest.fixture(scope="module")
def fanout_graph():
    return FanoutGraph()


def test_sample_fanouts(fanout_graph):
    np.random.seed(0)
    res = multihop.sample_fanout(
        fanout_graph, np.array([1]), metapath=[1, 1], fanouts=[3, 3], default_node=-1
    )
    assert len(res) == 3

    # initial nodes
    npt.assert_equal(res[0][0], np.array([1]))

    # 1st hop neighbors
    npt.assert_equal(res[0][1], np.array([2, 3, 2]))

    # 2nd hop neighbors
    npt.assert_equal(res[0][2], np.array([3, 3, 3, -1, -1, -1, 3, 3, 3]))

    # 1st hop weights
    npt.assert_equal(res[1][0], np.zeros(3))

    # 2nd hop weights
    npt.assert_equal(res[1][1], np.zeros(9))

    # 1st hop types
    npt.assert_equal(res[2][0], np.ones(3))

    # 2nd hop types
    npt.assert_equal(res[2][1], np.ones(9))


def test_sample_fanouts_empty_nodes(fanout_graph):
    np.random.seed(0)
    res = multihop.sample_fanout(
        fanout_graph, np.array([3]), metapath=[1, 1], fanouts=[2, 2], default_node=-1
    )
    assert len(res) == 3

    # initial nodes
    npt.assert_equal(res[0][0], np.array([3]))

    # 1st hop neighbors
    npt.assert_equal(res[0][1], np.array([-1, -1]))

    # 2nd hop neighbors
    npt.assert_equal(res[0][2], np.array([-1, -1, -1, -1]))

    # 1st hop weights
    npt.assert_equal(res[1][0], np.zeros(2))

    # 2nd hop weights
    npt.assert_equal(res[1][1], np.zeros(4))

    # 1st hop types
    npt.assert_equal(res[2][0], np.ones(2))

    # 2nd hop types
    npt.assert_equal(res[2][1], np.ones(4))
