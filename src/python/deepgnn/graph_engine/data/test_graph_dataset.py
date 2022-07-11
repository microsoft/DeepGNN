# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import tempfile
import os
import numpy as np
import pytest
import sys

from deepgnn.graph_engine._base import SamplingStrategy
from deepgnn.graph_engine.data.cora import CoraFull
from deepgnn.graph_engine.data.citeseer import CiteseerFull
from deepgnn.graph_engine.data.citation import CitationGraph, Cora
from deepgnn.graph_engine.data.ppi import PPI


def verify_nodes(g):
    batch_size = 10
    np.testing.assert_equal(g.NUM_NODES, g.node_count(types=0))
    for i in range(1000):
        nodes = g.sample_nodes(
            batch_size, node_types=0, strategy=SamplingStrategy.Random
        )
        assert (nodes > 0).all() and (nodes <= g.NUM_NODES).all()

        feat = g.node_features(
            nodes, np.array([[0, g.FEATURE_DIM]], dtype=np.int32), np.float32
        )
        assert feat.shape == (batch_size, g.FEATURE_DIM)
        np.testing.assert_array_equal(np.sum(feat, 1) > 0, [True] * batch_size)


def verify_train_test_nodes(data_dir, g, train_ratio):
    max_train_node_id = int(g.NUM_NODES * train_ratio)
    np.testing.assert_equal(max_train_node_id, g.node_count(types=0))
    np.testing.assert_equal(g.NUM_NODES - max_train_node_id, g.node_count(types=1))

    train_nodes = [int(i.strip()) for i in open(os.path.join(data_dir, "train.nodes"))]
    test_nodes = [int(i.strip()) for i in open(os.path.join(data_dir, "test.nodes"))]

    assert len(train_nodes) == max_train_node_id
    assert len(test_nodes) == g.NUM_NODES - max_train_node_id

    batch_size = 10
    # training node type: 0
    for i in range(1000):
        nodes = g.sample_nodes(
            batch_size, node_types=0, strategy=SamplingStrategy.Random
        )
        assert (nodes > 0).all() and (nodes <= max_train_node_id).all()
        np.testing.assert_array_equal(
            [nid in train_nodes for nid in nodes], [True] * batch_size
        )

    # testing node type: 1
    for i in range(1000):
        nodes = g.sample_nodes(
            batch_size, node_types=1, strategy=SamplingStrategy.Random
        )
        assert (nodes > max_train_node_id).all() and (nodes <= g.NUM_NODES).all()
        np.testing.assert_array_equal(
            [nid in test_nodes for nid in nodes], [True] * batch_size
        )


def verify_random_train_test_nodes(data_dir, g, train_ratio):
    train_cnt = int(g.NUM_NODES * train_ratio)
    np.testing.assert_equal(train_cnt, g.node_count(types=0))
    np.testing.assert_equal(g.NUM_NODES - train_cnt, g.node_count(types=1))
    train_nodes = [int(i.strip()) for i in open(os.path.join(data_dir, "train.nodes"))]
    test_nodes = [int(i.strip()) for i in open(os.path.join(data_dir, "test.nodes"))]

    np.testing.assert_equal(len(train_nodes), train_cnt)

    batch_size = 10
    # training node type: 0
    for i in range(1000):
        nodes = g.sample_nodes(
            batch_size, node_types=0, strategy=SamplingStrategy.Random
        )
        np.testing.assert_array_equal(
            [nid in train_nodes for nid in nodes], [True] * batch_size
        )

    # testing node type: 1
    for i in range(1000):
        nodes = g.sample_nodes(
            batch_size, node_types=1, strategy=SamplingStrategy.Random
        )
        np.testing.assert_array_equal(
            [nid in test_nodes for nid in nodes], [True] * batch_size
        )


def test_cora():
    g = CoraFull()
    assert g.NUM_CLASSES == 7
    verify_nodes(g)


def test_cora_train_ratio():
    tmp_dir = tempfile.TemporaryDirectory()
    data_dir = tmp_dir.name
    train_ratio = 0.8

    g = CoraFull(data_dir, train_ratio, random_selection=False)
    verify_train_test_nodes(data_dir, g, train_ratio)

    g = CoraFull(data_dir, train_ratio, random_selection=True)
    verify_random_train_test_nodes(data_dir, g, train_ratio)


def test_citeseer():
    g = CiteseerFull()
    assert g.NUM_CLASSES == 6
    verify_nodes(g)


def test_citeseer_train_ratio():
    tmp_dir = tempfile.TemporaryDirectory()
    data_dir = tmp_dir.name
    train_ratio = 0.8

    g = CiteseerFull(data_dir, train_ratio, random_selection=False)
    verify_train_test_nodes(data_dir, g, train_ratio)

    g = CiteseerFull(data_dir, train_ratio, random_selection=True)
    verify_random_train_test_nodes(data_dir, g, train_ratio)


def test_citation_graph():
    for name, train_cnt in [("cora", 140), ("citeseer", 120)]:
        g = CitationGraph(name)
        assert g.GRAPH_NAME == name

        batch_size = 10
        for _ in range(1000):
            nodes = g.sample_nodes(
                batch_size, node_types=0, strategy=SamplingStrategy.Random
            )
            np.testing.assert_array_equal((nodes >= 0).all(), [True] * batch_size)
            np.testing.assert_array_equal(
                (nodes < g.NUM_NODES).all(), [True] * batch_size
            )

            feat = g.node_features(
                nodes, np.array([[0, g.FEATURE_DIM]], dtype=np.int32), np.float32
            )
            assert feat.shape == (batch_size, g.FEATURE_DIM)
            np.testing.assert_array_equal(np.sum(feat, 1) > 0, [True] * batch_size)

        nodes = g.sample_nodes(
            train_cnt, node_types=0, strategy=SamplingStrategy.RandomWithoutReplacement
        )
        nodes.sort()
        np.testing.assert_equal(nodes, range(train_cnt))


def test_citation_graph_random_split():
    train_cnt = 140
    g = Cora(split="random")
    assert g.GRAPH_NAME == "cora"

    nodes = g.sample_nodes(
        train_cnt, node_types=0, strategy=SamplingStrategy.RandomWithoutReplacement
    )
    labels = g.node_features(nodes, np.array([[1, 1]], dtype=np.int32), np.float32)
    for c in range(g.NUM_CLASSES):
        # each class has 20 training nodes
        np.testing.assert_equal(np.sum(labels == c), 20)


def test_ppi():
    g = PPI()
    assert g.GRAPH_NAME == "ppi"
    np.testing.assert_equal(g.FEATURE_DIM, 50)
    np.testing.assert_equal(g.NUM_CLASSES, 121)
    np.testing.assert_equal(g.node_count(types=0), 44906)
    np.testing.assert_equal(g.node_count(types=1), 6514)
    np.testing.assert_equal(g.node_count(types=2), 5524)
    np.testing.assert_equal(g.edge_count(types=0), 1246382)
    np.testing.assert_equal(g.edge_count(types=1), 365966)

    # fmt: off
    def check_feat():
        feat_meta = np.array([[1, 50]], np.int32)
        nodes = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], np.int64)
        x = g.node_features(nodes, feat_meta, np.float32)
        x_row = x.sum(1)
        desired = [-6.3767824, -6.3767824, -6.3767824, -6.3767824, 3.6452491,
                   -6.3767824, 24.129421, -6.3767824, -6.3767824, -0.32132643]
        np.testing.assert_almost_equal(x_row, desired, decimal=5)

    def check_label():
        label_meta = np.array([[0, 121]], np.int32)
        nodes = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], np.int64)
        x = g.node_features(nodes, label_meta, np.float32)
        x_row = x.sum(1)
        desired = [34, 83, 33, 41, 72, 60, 60, 40, 31, 76]
        np.testing.assert_almost_equal(x_row, desired)
    # fmt: on
    check_feat()
    check_label()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, *sys.argv[1:]]))
