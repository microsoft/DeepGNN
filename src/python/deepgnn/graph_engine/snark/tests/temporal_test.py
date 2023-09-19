# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import numpy.testing as npt
import pytest

import deepgnn.graph_engine.snark.client as client
from deepgnn.graph_engine.snark.decoders import JsonDecoder
import deepgnn.graph_engine.snark.convert as convert
import deepgnn.graph_engine.snark.server as server
import deepgnn.graph_engine.snark.dispatcher as dispatcher
from util_test import find_free_port


def triangle_graph_json(folder):
    data = open(os.path.join(folder, "graph.json"), "w+")
    graph = [
        {
            "node_id": 9,
            "node_type": 0,
            "node_weight": 1,
            "float_feature": {
                "0": [
                    {"values": [123, 64, 12], "created_at": 13, "removed_at": 20},
                    {"values": [13, 14], "created_at": 20, "removed_at": 21},
                ],
            },
            "edge": [
                {
                    "src_id": 9,
                    "dst_id": 0,
                    "edge_type": 0,
                    "weight": 0.5,
                    "uint64_feature": {"0": [1, 2, 3]},
                    "float_feature": {},
                    "binary_feature": {},
                    "created_at": 1,
                    "removed_at": 5,
                }
            ],
            "created_at": 1,
        },
        {
            "node_id": 0,
            "node_type": 1,
            "node_weight": 1,
            "uint64_feature": {},
            "float_feature": {"0": [1], "1": [-0.03, -0.04]},
            "binary_feature": {"3": "abcd"},
            "edge": [
                {
                    "src_id": 0,
                    "dst_id": 5,
                    "edge_type": 1,
                    "weight": 1,
                    "float_feature": {"1": [3, 4]},
                    "created_at": 1,
                    "removed_at": 5,
                }
            ],
            "removed_at": 5,
        },
        {
            "node_id": 5,
            "node_type": 2,
            "node_weight": 1,
            "edge": [
                {
                    "src_id": 5,
                    "dst_id": 9,
                    "edge_type": 1,
                    "weight": 0.7,
                    "float_feature": {},
                }
            ],
        },
    ]
    for el in graph:
        json.dump(el, data)
        data.write("\n")
    data.flush()
    data.close()
    return data.name


@pytest.fixture(scope="module")
def triangle_graph_data():
    workdir = tempfile.TemporaryDirectory()
    data_name = triangle_graph_json(workdir.name)
    yield data_name
    workdir.cleanup()


# We'll use this class for deterministic partitioning
class Counter:
    def __init__(self):
        self.count = 0

    def __call__(self, x):
        self.count += 1
        return self.count % 2


@pytest.fixture(scope="module")
def multi_partition_graph_data():
    output = tempfile.TemporaryDirectory()
    data_name = triangle_graph_json(output.name)
    d = dispatcher.QueueDispatcher(
        Path(output.name), 2, Counter(), JsonDecoder(), watermark=0
    )
    convert.MultiWorkersConverter(
        graph_path=data_name,
        partition_count=1,
        output_dir=output.name,
        decoder=JsonDecoder(),
        dispatcher=d,
        watermark=0,
    ).convert()

    yield output.name


@pytest.fixture(scope="module", params=["inmemory", "distributed"])
def graph_dataset(multi_partition_graph_data, request):
    if request.param == "inmemory":
        yield client.MemoryGraph(multi_partition_graph_data, [0, 1])
    else:
        address = [f"localhost:{find_free_port()}", f"localhost:{find_free_port()}"]
        s1 = server.Server(multi_partition_graph_data, [0], address[0])
        s2 = server.Server(multi_partition_graph_data, [1], address[1])
        yield client.DistributedGraph(address)
        s1.reset()
        s2.reset()


def test_neighbor_sampling_graph_multiple_partitions(graph_dataset):
    ns, ws, tp, ts = graph_dataset.weighted_sample_neighbors(
        nodes=np.array([0, 1], dtype=np.int64),
        edge_types=1,
        count=2,
        seed=3,
        default_node=13,
        default_weight=-1,
        return_edge_created_ts=True,
    )

    npt.assert_array_equal(ns, [[5, 5], [13, 13]])
    npt.assert_array_equal(ws, [[1, 1], [-1, -1]])
    npt.assert_array_equal(tp, [[1, 1], [-1, -1]])
    npt.assert_array_equal(ts, [[-1, -1], [-1, -1]])


def test_neighbor_sampling_graph_multiple_partitions(graph_dataset):
    ns, ws, tp = graph_dataset.weighted_sample_neighbors(
        nodes=np.array([0, 1], dtype=np.int64),
        edge_types=1,
        count=2,
        seed=3,
        default_node=13,
        default_weight=-1,
    )

    npt.assert_array_equal(ns, [[5, 5], [13, 13]])
    npt.assert_array_equal(ws, [[1, 1], [-1, -1]])
    npt.assert_array_equal(tp, [[1, 1], [-1, -1]])


def test_neighbor_count_graph_nonmatching_edge_type(graph_dataset):
    output_node_counts = graph_dataset.neighbor_counts(
        nodes=np.array([9], dtype=np.int64), edge_types=100, timestamps=[1, 2, 1]
    )

    npt.assert_array_equal(output_node_counts, [0])


def test_neighbor_count_graph_nonexistent_node(graph_dataset):
    output_node_counts = graph_dataset.neighbor_counts(
        nodes=np.array([4], dtype=np.int64), edge_types=1, timestamps=[0]
    )

    npt.assert_array_equal(output_node_counts, [0])


def test_feature_values(graph_dataset):
    f = graph_dataset.node_features([9], [[0, 2]], dtype=np.float32, timestamps=[11])
    npt.assert_array_equal([[0, 0]], f)
    f = graph_dataset.node_features([9], [[0, 2]], dtype=np.float32, timestamps=[12])
    npt.assert_array_equal([[0, 0]], f)
    f = graph_dataset.node_features([9], [[0, 2]], dtype=np.float32, timestamps=[13])
    npt.assert_array_equal([[123, 64]], f)
    f = graph_dataset.node_features([9], [[0, 2]], dtype=np.float32, timestamps=[14])
    npt.assert_array_equal([[123, 64]], f)
    f = graph_dataset.node_features([9], [[0, 2]], dtype=np.float32, timestamps=[20])
    npt.assert_array_equal([[13, 14]], f)
    f = graph_dataset.node_features([9], [[0, 2]], dtype=np.float32, timestamps=[21])
    npt.assert_array_equal([[13, 14]], f)


if __name__ == "__main__":
    sys.exit(
        pytest.main(
            [__file__, "--junitxml", os.environ["XML_OUTPUT_FILE"], *sys.argv[1:]]
        )
    )
