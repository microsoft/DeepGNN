# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import json
import multiprocessing as mp
import os
import sys
import tempfile
import threading
import time
from pathlib import Path
import platform
from typing import List, Tuple

import numpy as np
import numpy.testing as npt
import pytest

import deepgnn.graph_engine.snark.client as client
from deepgnn.graph_engine.snark.decoders import JsonDecoder, EdgeListDecoder
import deepgnn.graph_engine.snark.server as server
import deepgnn.graph_engine.snark.convert as convert
import deepgnn.graph_engine.snark.dispatcher as dispatcher
import deepgnn.graph_engine.snark._lib as lib
from util_test import json_to_edge_list


def graph_with_sparse_features_json(folder):
    data = open(os.path.join(folder, "graph.json"), "w+")
    graph = [
        {
            "node_id": 9,
            "node_type": 0,
            "node_weight": 1,
            "float_feature": {"0": [0, 1], "3": [-0.01, -0.02]},
            "sparse_float_feature": {
                "2": {"coordinates": [5, 13], "values": [1.0, 2.13]}
            },
            "edge": [
                {
                    "src_id": 9,
                    "dst_id": 0,
                    "edge_type": 0,
                    "weight": 0.5,
                    # Edge and node features can have same feature ids
                    "uint64_feature": {"0": [1, 2, 3]},
                    # Sparse and dense features can not have same feature id
                    "sparse_uint8_feature": {
                        "7": {"coordinates": [5, 13], "values": [10, 255]}
                    },
                    "sparse_int8_feature": {
                        "2": {"coordinates": [7, 1], "values": [2, 1]}
                    },
                    "sparse_uint16_feature": {
                        "1": {"coordinates": [[12, 5, 3]], "values": [5]}
                    },
                    "sparse_int16_feature": {
                        "4": {"coordinates": [7, 3], "values": [255, 16]}
                    },
                    "sparse_uint32_feature": {
                        "3": {"coordinates": [9], "values": [4294967295]}
                    },
                    "sparse_int32_feature": {
                        "5": {"coordinates": [[5, 13], [7, 25]], "values": [-1, 1024]}
                    },
                    "sparse_uint64_feature": {
                        "12": {"coordinates": [2, 3], "values": [1, 8294967296]}
                    },
                    "sparse_int64_feature": {
                        "9": {
                            "coordinates": [4294967296, 8294967296],
                            "values": [4294967296, 23],
                        }
                    },
                    "sparse_double_feature": {
                        "10": {"coordinates": [0, 5], "values": [1.0, 2.13]}
                    },
                    "sparse_float16_feature": {
                        "11": {"coordinates": [1, -1], "values": [0.55, 0.33]}
                    },
                }
            ],
        },
        {
            "node_id": 0,
            "node_type": 1,
            "node_weight": 1,
            "float_feature": {"0": [1], "1": [-0.03, -0.04]},
            "sparse_float_feature": {
                "2": {"coordinates": [1, 3, 7], "values": [5.5, 6.5, 7.5]}
            },
            "edge": [
                {
                    "src_id": 0,
                    "dst_id": 5,
                    "edge_type": 1,
                    "weight": 1,
                    "float_feature": {"13": [3, 4]},
                    "sparse_uint16_feature": {
                        "1": {"coordinates": [[18, 15, 12]], "values": [1024]}
                    },
                    "sparse_int16_feature": {
                        "4": {"coordinates": [17, 13], "values": [2, 4]}
                    },
                }
            ],
        },
        {
            "node_id": 5,
            "node_type": 2,
            "node_weight": 1,
            "sparse_float_feature": {
                "2": {"coordinates": [4, 6], "values": [5.5, 6.89]}
            },
            "edge": [
                {
                    "src_id": 5,
                    "dst_id": 9,
                    "edge_type": 1,
                    "weight": 0.7,
                    "sparse_float_feature": {
                        "2": {"coordinates": [4, 6], "values": [5.5, 6.7]}
                    },
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
def graph_with_sparse_features(request):
    partition_count, decoder_type = request.param
    workdir = tempfile.TemporaryDirectory()
    data_name = graph_with_sparse_features_json(workdir.name)
    if decoder_type == EdgeListDecoder:
        json_name = data_name
        data_name = os.path.join(workdir.name, "graph.csv")
        json_to_edge_list(json_name, data_name)

    convert.MultiWorkersConverter(
        graph_path=data_name,
        partition_count=partition_count,
        output_dir=workdir.name,
        decoder=decoder_type(),
        skip_edge_sampler=True,
        skip_node_sampler=True,
    ).convert()
    yield client.MemoryGraph(workdir.name, list(range(partition_count)))
    workdir.cleanup()


@pytest.mark.parametrize(
    "graph_with_sparse_features",
    [(1, JsonDecoder), (1, EdgeListDecoder), (2, JsonDecoder), (2, EdgeListDecoder)],
    indirect=True,
)
def test_sanity_node_sparse_features(graph_with_sparse_features):
    indices, values, dimensions = graph_with_sparse_features.node_sparse_features(
        np.array([9], dtype=np.int64),
        features=np.array([2], dtype=np.int32),
        dtype=np.float32,
    )

    npt.assert_equal(indices, [[[0, 5], [0, 13]]])
    npt.assert_equal(dimensions, [1])
    npt.assert_allclose(values, [[1.0, 2.13]])


@pytest.mark.parametrize(
    "graph_with_sparse_features",
    [(1, JsonDecoder), (1, EdgeListDecoder), (2, JsonDecoder), (2, EdgeListDecoder)],
    indirect=True,
)
def test_multiple_nodes_sparse_features(graph_with_sparse_features):
    indices, values, dimensions = graph_with_sparse_features.node_sparse_features(
        np.array([9, 0, 5], dtype=np.int64),
        features=np.array([2], dtype=np.int32),
        dtype=np.float32,
    )

    npt.assert_equal(
        indices, [[[0, 5], [0, 13], [1, 1], [1, 3], [1, 7], [2, 4], [2, 6]]]
    )
    npt.assert_equal(dimensions, [1])
    npt.assert_allclose(values, [[1.0, 2.13, 5.5, 6.5, 7.5, 5.5, 6.89]])


@pytest.mark.parametrize(
    "graph_with_sparse_features",
    [(1, JsonDecoder), (1, EdgeListDecoder), (2, JsonDecoder), (2, EdgeListDecoder)],
    indirect=True,
)
def test_multiple_edges_sparse_features(graph_with_sparse_features):
    indices, values, dimensions = graph_with_sparse_features.edge_sparse_features(
        edge_src=np.array([9, 5, 0], dtype=np.int64),
        edge_dst=np.array([0, 9, 5], dtype=np.int64),
        edge_tp=np.array([0, 1, 1], dtype=np.int32),
        features=np.array([5], dtype=np.int32),
        dtype=np.int32,
    )

    npt.assert_equal(dimensions, [2])
    npt.assert_equal(indices, [[[0, 5, 13], [0, 7, 25]]])
    npt.assert_allclose(values, [[-1, 1024]])


@pytest.fixture(scope="module")
def multi_server_sparse_features_graph():
    workdir = tempfile.TemporaryDirectory()
    data_name = graph_with_sparse_features_json(workdir.name)
    convert.MultiWorkersConverter(
        graph_path=data_name,
        partition_count=2,
        output_dir=workdir.name,
        decoder=JsonDecoder(),
        skip_edge_sampler=True,
        skip_node_sampler=True,
    ).convert()
    s1 = server.Server(workdir.name, [0], hostname="localhost:1257")
    s2 = server.Server(workdir.name, [1], hostname="localhost:1258")

    yield client.DistributedGraph(["localhost:1257", "localhost:1258"])

    s1.reset()
    s2.reset()


def test_distributed_multiple_nodes_sparse_features(multi_server_sparse_features_graph):
    (
        indices,
        values,
        dimensions,
    ) = multi_server_sparse_features_graph.node_sparse_features(
        np.array([9, 0, 5], dtype=np.int64),
        features=np.array([2], dtype=np.int32),
        dtype=np.float32,
    )

    npt.assert_equal(
        indices, [[[0, 5], [0, 13], [1, 1], [1, 3], [1, 7], [2, 4], [2, 6]]]
    )
    npt.assert_equal(dimensions, [1])
    npt.assert_allclose(values, [[1.0, 2.13, 5.5, 6.5, 7.5, 5.5, 6.89]])


@pytest.mark.parametrize("multi_server_sparse_features_graph", [2], indirect=True)
def test_distributed_multiple_edges_sparse_features(multi_server_sparse_features_graph):
    (
        indices,
        values,
        dimensions,
    ) = multi_server_sparse_features_graph.edge_sparse_features(
        edge_src=np.array([9, 5, 0], dtype=np.int64),
        edge_dst=np.array([0, 9, 5], dtype=np.int64),
        edge_tp=np.array([3, 1, 1], dtype=np.int32),
        features=np.array([2], dtype=np.int32),
        dtype=np.float32,
    )

    npt.assert_equal(dimensions, [1])
    npt.assert_equal(indices, [[[1, 4], [1, 6]]])
    npt.assert_allclose(values, [[5.5, 6.7]])


@pytest.mark.parametrize("multi_server_sparse_features_graph", [2], indirect=True)
def test_distributed_multiple_edges_multiple_sparse_features(
    multi_server_sparse_features_graph,
):
    (
        indices,
        values,
        dimensions,
    ) = multi_server_sparse_features_graph.edge_sparse_features(
        edge_src=np.array([9, 5, 0], dtype=np.int64),
        edge_dst=np.array([0, 9, 5], dtype=np.int64),
        edge_tp=np.array([0, 1, 1], dtype=np.int32),
        features=np.array([4, 1], dtype=np.int32),
        dtype=np.int16,
    )

    npt.assert_equal(dimensions, [1, 3])
    assert len(indices) == 2
    npt.assert_equal(indices[0], [[0, 7], [0, 3], [2, 17], [2, 13]])
    npt.assert_equal(indices[1], [[0, 12, 5, 3], [2, 18, 15, 12]])
    npt.assert_equal(values, [[255, 16, 2, 4], [5, 1024]])


if __name__ == "__main__":
    sys.exit(
        pytest.main(
            [__file__, "--junitxml", os.environ["XML_OUTPUT_FILE"], *sys.argv[1:]]
        )
    )
