# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import json
import os
import sys
import tempfile
from pathlib import Path
from itertools import repeat

import numpy as np
import numpy.testing as npt
import pytest

import deepgnn.graph_engine.snark.client as client
from deepgnn.graph_engine.snark.decoders import JsonDecoder, EdgeListDecoder
import deepgnn.graph_engine.snark.server as server
import deepgnn.graph_engine.snark.convert as convert
import deepgnn.graph_engine.snark.dispatcher as dispatcher
from deepgnn.graph_engine.snark.converter.writers import BinaryWriter
from util_test import json_to_edge_list


nodes = [
    {
        "node_id": 9,
        "node_type": 0,
        "node_weight": 1,
        "float_feature": {"0": [0, 1], "3": [-0.01, -0.02]},
        "sparse_float_feature": {
            "1": {"coordinates": [], "values": []},
            "2": {"coordinates": [5, 13], "values": [1.0, 2.13]},
        },
        "edge": [],
    },
    {
        "node_id": 0,
        "node_type": 1,
        "node_weight": 1,
        "float_feature": {"0": [1], "1": [-0.03, -0.04]},
        "sparse_float_feature": {
            "2": {"coordinates": [1, 3, 7], "values": [5.5, 6.5, 7.5]}
        },
        "edge": [],
    },
    {
        "node_id": 5,
        "node_type": 2,
        "node_weight": 1,
        "sparse_float_feature": {"2": {"coordinates": [4, 6], "values": [5.5, 6.89]}},
        "edge": [],
    },
]

edges = [
    {
        "src_id": 9,
        "dst_id": 0,
        "edge_type": 0,
        "weight": 0.5,
        # Edge and node features can have same feature ids
        "uint64_feature": {"0": [1, 2, 3]},
        # Sparse and dense features can not have same feature id
        "sparse_uint8_feature": {"7": {"coordinates": [5, 13], "values": [10, 255]}},
        "sparse_int8_feature": {"2": {"coordinates": [7, 1], "values": [2, 1]}},
        "sparse_uint16_feature": {"1": {"coordinates": [[12, 5, 3]], "values": [5]}},
        "sparse_int16_feature": {"4": {"coordinates": [7, 3], "values": [255, 16]}},
        "sparse_uint32_feature": {"3": {"coordinates": [9], "values": [4294967295]}},
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
        "sparse_double_feature": {"10": {"coordinates": [0, 5], "values": [1.0, 2.13]}},
        "sparse_float16_feature": {
            "11": {"coordinates": [1, -1], "values": [0.55, 0.33]}
        },
    },
    {
        "src_id": 0,
        "dst_id": 5,
        "edge_type": 1,
        "weight": 1,
        "float_feature": {"13": [3, 4]},
        "sparse_uint16_feature": {
            "1": {"coordinates": [[18, 15, 12]], "values": [1024]}
        },
        "sparse_int16_feature": {"4": {"coordinates": [17, 13], "values": [2, 4]}},
    },
    {
        "src_id": 5,
        "dst_id": 9,
        "edge_type": 1,
        "weight": 0.7,
        "sparse_float_feature": {"2": {"coordinates": [4, 6], "values": [5.5, 6.7]}},
    },
]


def graph_with_sparse_features_json(folder):
    data = open(os.path.join(folder, "graph.json"), "w+")
    nodes[0]["edge"] = [edges[0]]
    nodes[1]["edge"] = [edges[1]]
    nodes[2]["edge"] = [edges[2]]
    graph = [
        nodes[0],
        nodes[1],
        nodes[2],
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
    partitions = list(
        zip(repeat(workdir.name, partition_count), range(partition_count))
    )
    yield client.MemoryGraph(workdir.name, partitions)

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
        features=np.array([13], dtype=np.int32),
        dtype=np.int32,
    )

    npt.assert_equal(dimensions, [0])
    assert len(indices) == 1
    assert len(values) == 1


@pytest.mark.parametrize(
    "graph_with_sparse_features",
    [(1, JsonDecoder), (1, EdgeListDecoder), (2, JsonDecoder), (2, EdgeListDecoder)],
    indirect=True,
)
def test_multiple_edges_empty_sparse_features(graph_with_sparse_features):
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
def multi_server_sparse_features_graph(request):
    decoder_type = request.param
    workdir = tempfile.TemporaryDirectory()
    data_name = graph_with_sparse_features_json(workdir.name)
    if decoder_type == EdgeListDecoder:
        json_name = data_name
        data_name = os.path.join(workdir.name, "graph.csv")
        json_to_edge_list(json_name, data_name)
    convert.MultiWorkersConverter(
        graph_path=data_name,
        partition_count=2,
        output_dir=workdir.name,
        decoder=decoder_type(),
        skip_edge_sampler=True,
        skip_node_sampler=True,
    ).convert()
    s1 = server.Server(workdir.name, [(workdir.name, 0)], hostname="localhost:1257")
    s2 = server.Server(workdir.name, [(workdir.name, 1)], hostname="localhost:1258")

    yield client.DistributedGraph(["localhost:1257", "localhost:1258"])

    s1.reset()
    s2.reset()


@pytest.mark.parametrize(
    "multi_server_sparse_features_graph",
    [JsonDecoder, EdgeListDecoder],
    indirect=True,
)
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


@pytest.mark.parametrize(
    "multi_server_sparse_features_graph",
    [JsonDecoder, EdgeListDecoder],
    indirect=True,
)
def test_distributed_node_with_empty_sparse_features(
    multi_server_sparse_features_graph,
):
    (
        indices,
        values,
        dimensions,
    ) = multi_server_sparse_features_graph.node_sparse_features(
        np.array([9], dtype=np.int64),
        features=np.array([1], dtype=np.int32),
        dtype=np.float32,
    )

    assert len(indices) == 1
    assert len(values) == 1
    assert dimensions == [0]


@pytest.mark.parametrize(
    "multi_server_sparse_features_graph",
    [JsonDecoder, EdgeListDecoder],
    indirect=True,
)
def test_distributed_node_with_wrong_id_sparse_features(
    multi_server_sparse_features_graph,
):
    (
        indices,
        values,
        dimensions,
    ) = multi_server_sparse_features_graph.node_sparse_features(
        np.array([9, 0], dtype=np.int64),
        features=np.array([0, 3], dtype=np.int32),
        dtype=np.float32,
    )

    assert len(indices) == 2
    assert len(values) == 2
    npt.assert_equal(dimensions, [0, 0])


@pytest.mark.parametrize(
    "multi_server_sparse_features_graph",
    [JsonDecoder, EdgeListDecoder],
    indirect=True,
)
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


@pytest.mark.parametrize(
    "multi_server_sparse_features_graph",
    [JsonDecoder, EdgeListDecoder],
    indirect=True,
)
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
        features=np.array([4, 13, 1], dtype=np.int32),
        dtype=np.int16,
    )

    npt.assert_equal(dimensions, [1, 0, 3])
    assert len(indices) == 3
    npt.assert_equal(indices[0], [[0, 7], [0, 3], [2, 17], [2, 13]])
    npt.assert_equal(indices[1], [])
    npt.assert_equal(indices[2], [[0, 12, 5, 3], [2, 18, 15, 12]])
    npt.assert_equal(values, [[255, 16, 2, 4], [], [5, 1024]])


# We'll use this class for deterministic partitioning
class Counter:
    def __init__(self):
        self.count = 0

    def __call__(self, x):
        self.count += 1
        return self.count % 2


@pytest.fixture(scope="module")
def multi_partition_graph_data(request):
    output = tempfile.TemporaryDirectory()
    if request.param == "original":
        data_name = graph_with_sparse_features_json(output.name)
        d = dispatcher.QueueDispatcher(Path(output.name), 2, Counter(), JsonDecoder())
        convert.MultiWorkersConverter(
            graph_path=data_name,
            partition_count=2,
            output_dir=output.name,
            decoder=JsonDecoder(),
            dispatcher=d,
            skip_edge_sampler=True,
            skip_node_sampler=True,
        ).convert()
    elif request.param == "nodes_p0":
        partitions = [
            nodes,
            edges,
        ]
        write_multi_binary(output.name, partitions)
    elif request.param == "nodes_p1":
        partitions = [
            edges,
            nodes,
        ]
        write_multi_binary(output.name, partitions)
    yield output.name


param = ["original", "nodes_p0", "nodes_p1"]


def linearize(value):
    def get_features(item):
        output = []
        for feature_key, feature_dict in item.items():
            if "feature" not in feature_key:
                continue
            for key, v in feature_dict.items():
                while int(key) + 1 > len(output):
                    output.append(None)
                if feature_key == "binary_feature":
                    vv = v
                elif "sparse" in feature_key:
                    vv = (
                        np.array(v["coordinates"], dtype=np.int64),
                        np.array(
                            v["values"],
                            dtype=JsonDecoder.convert_map[
                                feature_key.replace("sparse_", "")
                            ],
                        ),
                    )
                else:
                    vv = np.array(v, dtype=JsonDecoder.convert_map[feature_key])
                output[int(key)] = vv
        return output

    if "node_id" in value:
        return (
            value["node_id"],
            -1,
            value["node_type"],
            value["node_weight"],
            None,
            None,
            get_features(value),
        )
    else:
        return (
            value["src_id"],
            value["dst_id"],
            value["edge_type"],
            value["weight"],
            None,
            None,
            get_features(value),
        )


def write_multi_binary(output_dir, partitions):
    for i, p in enumerate(partitions):
        writer = BinaryWriter(output_dir, i)
        for v in p:
            writer.add([linearize(v)])
        writer.close()
    content = {
        "binary_data_version": "v2",  # converter version
        "node_count": 3,
        "edge_count": 3,
        "node_type_count": 3,
        "edge_type_count": 2,
        "node_feature_count": 15,
        "edge_feature_count": 15,
        "partitions": {
            f"{i}": {"node_weight": [3, 3, 3], "edge_weight": [2, 2]} for i in range(2)
        },
        "node_count_per_type": [1, 1, 1],
        "edge_count_per_type": [1, 2],
        "watermark": -1,
    }
    with open(os.path.join(output_dir, "meta.json"), "w+") as f:
        f.write(json.dumps(content))


@pytest.mark.parametrize("multi_partition_graph_data", param, indirect=True)
def test_sparse_node_features_graph_multiple_partitions(multi_partition_graph_data):
    cl = client.MemoryGraph(
        multi_partition_graph_data,
        [(multi_partition_graph_data, 0), (multi_partition_graph_data, 1)],
    )
    indices, values, dimensions = cl.node_sparse_features(
        np.array([9, 0, 5], dtype=np.int64),
        features=np.array([2], dtype=np.int32),
        dtype=np.float32,
    )

    npt.assert_equal(
        indices, [[[0, 5], [0, 13], [1, 1], [1, 3], [1, 7], [2, 4], [2, 6]]]
    )
    npt.assert_equal(dimensions, [1])
    npt.assert_allclose(values, [[1.0, 2.13, 5.5, 6.5, 7.5, 5.5, 6.89]])


@pytest.mark.parametrize("multi_partition_graph_data", param, indirect=True)
def test_sparse_edge_features_graph_multiple_partitions(multi_partition_graph_data):
    cl = client.MemoryGraph(
        multi_partition_graph_data,
        [(multi_partition_graph_data, 0), (multi_partition_graph_data, 1)],
    )
    indices, values, dimensions = cl.edge_sparse_features(
        edge_src=np.array([9, 5, 0], dtype=np.int64),
        edge_dst=np.array([0, 9, 5], dtype=np.int64),
        edge_tp=np.array([0, 1, 1], dtype=np.int32),
        features=np.array([5], dtype=np.int32),
        dtype=np.int32,
    )

    npt.assert_equal(dimensions, [2])
    npt.assert_equal(indices, [[[0, 5, 13], [0, 7, 25]]])
    npt.assert_allclose(values, [[-1, 1024]])


@pytest.mark.parametrize("multi_partition_graph_data", param, indirect=True)
def test_remote_client_sparse_node_features_graph_multiple_partitions(
    multi_partition_graph_data,
):
    address = ["localhost:1336", "localhost:1337"]
    s1 = server.Server(
        multi_partition_graph_data, [(multi_partition_graph_data, 0)], address[0]
    )
    s2 = server.Server(
        multi_partition_graph_data, [(multi_partition_graph_data, 1)], address[1]
    )
    cl = client.DistributedGraph(address)
    indices, values, dimensions = cl.node_sparse_features(
        np.array([9, 0, 5], dtype=np.int64),
        features=np.array([2], dtype=np.int32),
        dtype=np.float32,
    )

    npt.assert_equal(
        indices, [[[0, 5], [0, 13], [1, 1], [1, 3], [1, 7], [2, 4], [2, 6]]]
    )
    npt.assert_equal(dimensions, [1])
    npt.assert_allclose(values, [[1.0, 2.13, 5.5, 6.5, 7.5, 5.5, 6.89]])
    s1.reset()
    s2.reset()


@pytest.mark.parametrize("multi_partition_graph_data", param, indirect=True)
def test_remote_client_sparse_edge_features_graph_multiple_partitions(
    multi_partition_graph_data,
):
    address = ["localhost:1338", "localhost:1339"]
    s1 = server.Server(
        multi_partition_graph_data, [(multi_partition_graph_data, 0)], address[0]
    )
    s2 = server.Server(
        multi_partition_graph_data, [(multi_partition_graph_data, 1)], address[1]
    )
    cl = client.DistributedGraph(address)
    indices, values, dimensions = cl.edge_sparse_features(
        edge_src=np.array([9, 5, 0], dtype=np.int64),
        edge_dst=np.array([0, 9, 5], dtype=np.int64),
        edge_tp=np.array([0, 1, 1], dtype=np.int32),
        features=np.array([5], dtype=np.int32),
        dtype=np.int32,
    )

    npt.assert_equal(dimensions, [2])
    npt.assert_equal(indices, [[[0, 5, 13], [0, 7, 25]]])
    npt.assert_allclose(values, [[-1, 1024]])
    s1.reset()
    s2.reset()


if __name__ == "__main__":
    sys.exit(
        pytest.main(
            [__file__, "--junitxml", os.environ["XML_OUTPUT_FILE"], *sys.argv[1:]]
        )
    )
