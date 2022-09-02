# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import json
import os
import struct
import sys
import tempfile
from pathlib import Path

import pytest
import numpy as np
import numpy.testing as npt

import deepgnn.graph_engine.snark.convert as convert
from deepgnn.graph_engine.snark.decoders import JsonDecoder, TsvDecoder
from deepgnn.graph_engine.snark.dispatcher import QueueDispatcher
from deepgnn.graph_engine.snark.meta import BINARY_DATA_VERSION


def triangle_graph_json(folder):
    data = open(os.path.join(folder, "graph.json"), "w+")
    graph = [
        {
            "node_id": 9,
            "node_type": 0,
            "node_weight": 1,
            "uint64_feature": {},
            "float_feature": {"0": [0, 1], "1": [-0.01, -0.02]},
            "binary_feature": {},
            "edge": [
                {
                    "src_id": 9,
                    "dst_id": 0,
                    "edge_type": 0,
                    "weight": 0.5,
                    "uint64_feature": {"0": [1, 2, 3]},
                    "float_feature": {},
                    "binary_feature": {},
                }
            ],
        },
        {
            "node_id": 0,
            "node_type": 1,
            "node_weight": 1,
            "uint64_feature": {},
            "float_feature": {"0": [1], "1": [-0.03, -0.04]},
            "binary_feature": {},
            "edge": [
                {
                    "src_id": 0,
                    "dst_id": 5,
                    "edge_type": 1,
                    "weight": 1,
                    "uint64_feature": {},
                    "float_feature": {"1": [3, 4]},
                    "binary_feature": {},
                }
            ],
        },
        {
            "node_id": 5,
            "node_type": 2,
            "node_weight": 1,
            "uint64_feature": {},
            "float_feature": {"0": [1, 1], "1": [-0.05, -0.06]},
            "binary_feature": {},
            "edge": [
                {
                    "src_id": 5,
                    "dst_id": 9,
                    "edge_type": 1,
                    "weight": 0.7,
                    "uint64_feature": {},
                    "float_feature": {},
                    "binary_feature": {"0": "hello"},
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


def triangle_graph_tsv(folder):
    data = open(os.path.join(folder, "graph.tsv"), "w+")
    data.write("9\t0\t1\tf:0 1;f:-0.01 -0.02\t0,0,0.5,u64:1 2 3\n")
    data.write("0\t1\t1\tf:1;f:-0.03 -0.04\t5,1,1,;f:3 4\n")
    data.write("5\t2\t1\tf:1 1;f:-0.05 -0.06\t9,1,0.7,b:hello\n")
    data.flush()
    data.close()
    return data.name


@pytest.fixture(scope="module")
def triangle_graph(request):
    workdir = tempfile.TemporaryDirectory()
    if request.param == JsonDecoder:
        data_name = triangle_graph_json(workdir.name)
    elif request.param == TsvDecoder:
        data_name = triangle_graph_tsv(workdir.name)
    else:
        raise ValueError("Unsupported format.")

    yield data_name, request.param
    workdir.cleanup()


param = [JsonDecoder, TsvDecoder]


@pytest.mark.parametrize("triangle_graph", param, indirect=True)
def test_sanity_node_map(triangle_graph):
    output = tempfile.TemporaryDirectory()
    data_name, decoder = triangle_graph
    convert.MultiWorkersConverter(
        graph_path=data_name,
        partition_count=1,
        output_dir=output.name,
        decoder=decoder(),
    ).convert()

    with open("{}/node_{}_{}.map".format(output.name, 0, 0), "rb") as nm:
        expected_size = 3 * (2 * 8 + 4)
        result = nm.read(expected_size + 8)
        assert len(result) == expected_size
        assert result[0:8] == (9).to_bytes(8, byteorder=sys.byteorder)
        assert result[8:16] == (0).to_bytes(8, byteorder=sys.byteorder)
        assert result[16:20] == (0).to_bytes(4, byteorder=sys.byteorder)
        assert result[20:28] == (0).to_bytes(8, byteorder=sys.byteorder)
        assert result[28:36] == (1).to_bytes(8, byteorder=sys.byteorder)
        assert result[36:40] == (1).to_bytes(4, byteorder=sys.byteorder)
        assert result[40:48] == (5).to_bytes(8, byteorder=sys.byteorder)
        assert result[48:56] == (2).to_bytes(8, byteorder=sys.byteorder)
        assert result[56:60] == (2).to_bytes(4, byteorder=sys.byteorder)


@pytest.mark.parametrize("triangle_graph", param, indirect=True)
def test_sanity_node_index(triangle_graph):
    output = tempfile.TemporaryDirectory()
    data_name, decoder = triangle_graph
    convert.MultiWorkersConverter(
        graph_path=data_name,
        partition_count=1,
        output_dir=output.name,
        decoder=decoder(),
    ).convert()
    with open("{}/node_{}_{}.index".format(output.name, 0, 0), "rb") as ni:
        expected_size = 3 * 8 + 8
        result = ni.read(expected_size + 8)
        assert len(result) == expected_size
        assert result[0:8] == (0).to_bytes(8, byteorder=sys.byteorder)
        assert result[8:16] == (2).to_bytes(8, byteorder=sys.byteorder)
        assert result[16:24] == (4).to_bytes(8, byteorder=sys.byteorder)
        assert result[24:32] == (6).to_bytes(8, byteorder=sys.byteorder)


@pytest.mark.parametrize("triangle_graph", param, indirect=True)
def test_sanity_node_feature_index(triangle_graph):
    output = tempfile.TemporaryDirectory()
    data_name, decoder = triangle_graph
    convert.MultiWorkersConverter(
        graph_path=data_name,
        partition_count=1,
        output_dir=output.name,
        decoder=decoder(),
    ).convert()
    with open("{}/node_features_{}_{}.index".format(output.name, 0, 0), "rb") as ni:
        expected_size = (
            3 * 8 * 2 + 8
        )  # 3 nodes, 2 features each with 2 float + 8 as final close
        result = ni.read(expected_size + 1)
        assert len(result) == expected_size
        assert result[0:8] == (0).to_bytes(8, byteorder=sys.byteorder)
        assert result[8:16] == (8).to_bytes(8, byteorder=sys.byteorder)
        assert result[16:24] == (16).to_bytes(8, byteorder=sys.byteorder)
        assert result[24:32] == (20).to_bytes(8, byteorder=sys.byteorder)
        assert result[32:40] == (28).to_bytes(8, byteorder=sys.byteorder)
        assert result[40:48] == (36).to_bytes(8, byteorder=sys.byteorder)
        assert result[48:56] == (44).to_bytes(8, byteorder=sys.byteorder)


@pytest.mark.parametrize("triangle_graph", param, indirect=True)
def test_sanity_neighbors_index(triangle_graph):
    output = tempfile.TemporaryDirectory()
    data_name, decoder = triangle_graph
    convert.MultiWorkersConverter(
        graph_path=data_name,
        partition_count=1,
        output_dir=output.name,
        decoder=decoder(),
    ).convert()
    with open("{}/neighbors_{}_{}.index".format(output.name, 0, 0), "rb") as ni:
        expected_size = 3 * 8 + 8  # 3 nodes + 8 as final close
        result = ni.read(expected_size + 1)
        assert len(result) == expected_size
        assert result[0:8] == (0).to_bytes(8, byteorder=sys.byteorder)
        assert result[8:16] == (1).to_bytes(8, byteorder=sys.byteorder)
        assert result[16:24] == (2).to_bytes(8, byteorder=sys.byteorder)
        assert result[24:32] == (3).to_bytes(8, byteorder=sys.byteorder)


@pytest.mark.parametrize("triangle_graph", param, indirect=True)
def test_sanity_edge_index(triangle_graph):
    output = tempfile.TemporaryDirectory()
    data_name, decoder = triangle_graph
    convert.MultiWorkersConverter(
        graph_path=data_name,
        partition_count=1,
        output_dir=output.name,
        decoder=decoder(),
    ).convert()
    with open("{}/edge_{}_{}.index".format(output.name, 0, 0), "rb") as ei:
        expected_size = 4 * 24  # 3 nodes + last line as final close
        result = ei.read(expected_size + 100)
        assert len(result) == expected_size
        assert result[0:8] == (0).to_bytes(8, byteorder=sys.byteorder)
        assert result[8:16] == (0).to_bytes(8, byteorder=sys.byteorder)
        assert result[16:20] == (0).to_bytes(4, byteorder=sys.byteorder)
        assert result[20:24] == struct.pack("f", 0.5)

        assert result[24:32] == (5).to_bytes(8, byteorder=sys.byteorder)
        assert result[32:40] == (1).to_bytes(8, byteorder=sys.byteorder)
        assert result[40:44] == (1).to_bytes(4, byteorder=sys.byteorder)
        assert result[44:48] == struct.pack("f", 1.0)

        assert result[48:56] == (9).to_bytes(8, byteorder=sys.byteorder)
        assert result[56:64] == (3).to_bytes(8, byteorder=sys.byteorder)
        assert result[64:68] == (1).to_bytes(4, byteorder=sys.byteorder)
        assert result[68:72] == struct.pack("f", 0.7)

        assert result[72:80] == (0).to_bytes(8, byteorder=sys.byteorder)
        assert result[80:88] == (4).to_bytes(8, byteorder=sys.byteorder)
        assert result[88:92] == (0).to_bytes(4, byteorder=sys.byteorder)
        assert result[92:96] == struct.pack("f", -1)


@pytest.mark.parametrize("triangle_graph", param, indirect=True)
def test_sanity_edge_features_index(triangle_graph):
    output = tempfile.TemporaryDirectory()
    data_name, decoder = triangle_graph
    convert.MultiWorkersConverter(
        graph_path=data_name,
        partition_count=1,
        output_dir=output.name,
        decoder=decoder(),
    ).convert()
    with open("{}/edge_features_{}_{}.index".format(output.name, 0, 0), "rb") as ni:
        expected_values = [0, 24, 24, 32, 37]

        expected_size = len(expected_values) * 8
        result = ni.read(expected_size)
        assert len(result) == expected_size
        for i in range(len(expected_values)):
            assert result[i * 8 : (i + 1) * 8] == (expected_values[i]).to_bytes(
                8, byteorder=sys.byteorder
            )


@pytest.mark.parametrize("triangle_graph", param, indirect=True)
def test_sanity_edge_features_data(triangle_graph):
    output = tempfile.TemporaryDirectory()
    data_name, decoder = triangle_graph
    convert.MultiWorkersConverter(
        graph_path=data_name,
        partition_count=1,
        output_dir=output.name,
        decoder=decoder(),
    ).convert()
    with open("{}/edge_features_{}_{}.data".format(output.name, 0, 0), "rb") as ni:
        expected_size = 37  # last value in edge_features_index
        result = ni.read(expected_size + 1)
        assert len(result) == expected_size
        assert result[0:8] == (1).to_bytes(8, byteorder=sys.byteorder)
        assert result[8:16] == (2).to_bytes(8, byteorder=sys.byteorder)
        assert result[16:24] == (3).to_bytes(8, byteorder=sys.byteorder)

        assert result[24:28] == struct.pack("=f", 3)
        assert result[28:32] == struct.pack("=f", 4)

        assert result[32:37] == bytes("hello", "utf-8")


@pytest.mark.parametrize("triangle_graph", param, indirect=True)
def test_sanity_metadata(triangle_graph):
    output = tempfile.TemporaryDirectory()
    data_name, decoder = triangle_graph
    convert.MultiWorkersConverter(
        graph_path=data_name,
        partition_count=1,
        output_dir=output.name,
        decoder=decoder(),
    ).convert()
    with open("{}/meta.txt".format(output.name), "r") as ni:
        result = ni.readlines()

        assert len(result) == 19
        assert result[0].strip() == BINARY_DATA_VERSION
        assert int(result[1]) == 3
        assert int(result[2]) == 3
        assert int(result[3]) == 3
        assert int(result[4]) == 2
        assert int(result[5]) == 2
        assert int(result[6]) == 2

        # partition information
        assert int(result[7]) == 1
        assert int(result[8]) == 0
        assert float(result[9]) == 1
        assert float(result[10]) == 1
        assert float(result[11]) == 1
        assert float(result[12]) == 0.5
        assert float(result[13]) == 1.7

        # type counts
        assert int(result[14]) == 1
        assert int(result[15]) == 1
        assert int(result[16]) == 1
        assert int(result[17]) == 1
        assert int(result[18]) == 2


@pytest.mark.parametrize("triangle_graph", param, indirect=True)
def test_edge_alias_tables(triangle_graph):
    output = tempfile.TemporaryDirectory()
    data_name, decoder = triangle_graph

    class Counter:
        def __init__(self):
            self.count = -1

        def __call__(self, x):
            self.count += 1
            return self.count % 2

    d = QueueDispatcher(Path(output.name), 2, Counter(), decoder())
    convert.MultiWorkersConverter(
        graph_path=data_name,
        partition_count=2,
        output_dir=output.name,
        decoder=decoder(),
        dispatcher=d,
    ).convert()
    with open("{}/edge_0_0.alias".format(output.name), "rb") as ea:
        expected_size = 36  # Only 1 record
        result = ea.read(expected_size + 1)
        assert len(result) == expected_size
        assert result[0:8] == (9).to_bytes(8, byteorder=sys.byteorder)
        assert result[8:16] == (0).to_bytes(8, byteorder=sys.byteorder)
        assert result[16:24] == (0).to_bytes(8, byteorder=sys.byteorder)
        assert result[24:32] == (0).to_bytes(8, byteorder=sys.byteorder)
        assert result[32:36] == struct.pack("=f", 1.0)

    with open("{}/edge_1_0.alias".format(output.name), "rb") as ea:
        expected_size = 36  # Only 1 record
        result = ea.read(expected_size + 1)
        assert len(result) == expected_size
        assert result[0:8] == (5).to_bytes(8, byteorder=sys.byteorder)
        assert result[8:16] == (9).to_bytes(8, byteorder=sys.byteorder)
        assert result[16:24] == (0).to_bytes(8, byteorder=sys.byteorder)
        assert result[24:32] == (0).to_bytes(8, byteorder=sys.byteorder)
        assert result[32:36] == struct.pack("=f", 1.0)

    with open("{}/edge_1_1.alias".format(output.name), "rb") as ea:
        expected_size = 36  # Only 1 record
        result = ea.read(expected_size + 1)
        assert len(result) == expected_size
        assert result[0:8] == (0).to_bytes(8, byteorder=sys.byteorder)
        assert result[8:16] == (5).to_bytes(8, byteorder=sys.byteorder)
        assert result[16:24] == (0).to_bytes(8, byteorder=sys.byteorder)
        assert result[24:32] == (0).to_bytes(8, byteorder=sys.byteorder)
        assert result[32:36] == struct.pack("=f", 1.0)
    assert os.path.getsize("{}/edge_0_1.alias".format(output.name)) == 0


@pytest.mark.parametrize("triangle_graph", param, indirect=True)
def test_node_alias_tables(triangle_graph):
    output = tempfile.TemporaryDirectory()
    data_name, decoder = triangle_graph

    class Counter:
        def __init__(self):
            self.count = -1

        def __call__(self, x):
            self.count += 1
            return self.count % 2

    d = QueueDispatcher(Path(output.name), 2, Counter(), decoder())
    convert.MultiWorkersConverter(
        graph_path=data_name,
        partition_count=2,
        output_dir=output.name,
        decoder=decoder(),
        dispatcher=d,
    ).convert()

    with open("{}/node_0_0.alias".format(output.name), "rb") as ea:
        expected_size = 20  # Only 1 record
        result = ea.read(expected_size + 1)
        assert len(result) == expected_size
        assert result[0:8] == (9).to_bytes(8, byteorder=sys.byteorder)
        assert result[8:16] == (0).to_bytes(8, byteorder=sys.byteorder)
        assert result[16:20] == struct.pack("=f", 1.0)

    with open("{}/node_1_1.alias".format(output.name), "rb") as ea:
        expected_size = 20  # Only 1 record
        result = ea.read(expected_size + 1)
        assert len(result) == expected_size
        assert result[0:8] == (0).to_bytes(8, byteorder=sys.byteorder)
        assert result[8:16] == (0).to_bytes(8, byteorder=sys.byteorder)
        assert result[16:20] == struct.pack("=f", 1.0)

    with open("{}/node_2_0.alias".format(output.name), "rb") as ea:
        expected_size = 20  # Only 1 record
        result = ea.read(expected_size + 1)
        assert len(result) == expected_size
        assert result[0:8] == (5).to_bytes(8, byteorder=sys.byteorder)
        assert result[8:16] == (0).to_bytes(8, byteorder=sys.byteorder)
        assert result[16:20] == struct.pack("=f", 1.0)

    assert os.path.getsize("{}/node_0_1.alias".format(output.name)) == 0
    assert os.path.getsize("{}/node_2_1.alias".format(output.name)) == 0
    assert os.path.getsize("{}/node_1_0.alias".format(output.name)) == 0


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
                        "1": {"coordinates": [7, 1], "values": [2, 1]}
                    },
                    "sparse_uint16_feature": {
                        "2": {"coordinates": [[12, 5, 3]], "values": [65535]}
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
            "sparse_float_feature": {"2": {"coordinates": [1, 3, 7], "values": [5.5]}},
            "edge": [
                {
                    "src_id": 0,
                    "dst_id": 5,
                    "edge_type": 1,
                    "weight": 1,
                    "float_feature": {"1": [3, 4]},
                }
            ],
        },
        {
            "node_id": 5,
            "node_type": 2,
            "node_weight": 1,
            "sparse_float_feature": {
                "2": {"coordinates": [1, 3], "values": [5.5, 6.89]}
            },
            "edge": [
                {
                    "src_id": 5,
                    "dst_id": 9,
                    "edge_type": 1,
                    "weight": 0.7,
                    "sparse_float_feature": {
                        "2": {"coordinates": [1, 3], "values": [5.5, 6.7]}
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
    workdir = tempfile.TemporaryDirectory()
    if request.param == JsonDecoder:
        data_name = graph_with_sparse_features_json(workdir.name)
    else:
        raise ValueError("Unsupported format.")

    yield data_name, request.param
    workdir.cleanup()


@pytest.mark.parametrize("graph_with_sparse_features", [JsonDecoder], indirect=True)
def test_sanity_node_sparse_features_index(graph_with_sparse_features):
    output = tempfile.TemporaryDirectory()
    data_name, decoder = graph_with_sparse_features
    convert.MultiWorkersConverter(
        graph_path=data_name,
        partition_count=1,
        output_dir=output.name,
        decoder=decoder(),
    ).convert()
    with open("{}/node_features_{}_{}.index".format(output.name, 0, 0), "rb") as ni:
        expected_size = 88
        result = ni.read(expected_size + 1)
        assert len(result) == expected_size
        actual = []
        for i in range(11):
            actual.append(int.from_bytes(result[i * 8 : (i + 1) * 8], sys.byteorder))
        assert actual[0:6] == [0, 8, 8, 40, 48, 52]
        assert actual[6:11] == [60, 96, 96, 96, 128]


@pytest.mark.parametrize("graph_with_sparse_features", [JsonDecoder], indirect=True)
def test_sanity_node_sparse_features_data(graph_with_sparse_features):
    output = tempfile.TemporaryDirectory()
    data_name, decoder = graph_with_sparse_features
    convert.MultiWorkersConverter(
        graph_path=data_name,
        partition_count=1,
        output_dir=output.name,
        decoder=decoder(),
    ).convert()
    with open("{}/node_features_{}_{}.data".format(output.name, 0, 0), "rb") as nfd:
        expected_size = 128  # last value in edge_features_index
        result = nfd.read(expected_size + 1)
        assert len(result) == expected_size
        npt.assert_equal(np.frombuffer(result[0:8], dtype=np.float32), [0, 1])

        assert int.from_bytes(result[8:12], sys.byteorder) == 2
        assert int.from_bytes(result[12:16], sys.byteorder) == 1
        npt.assert_equal(np.frombuffer(result[16:32], dtype=np.int64), [5, 13])
        npt.assert_almost_equal(
            np.frombuffer(result[32:40], dtype=np.float32), [1.0, 2.13]
        )

        npt.assert_almost_equal(
            np.frombuffer(result[40:48], dtype=np.float32), [-0.01, -0.02]
        )
        npt.assert_almost_equal(np.frombuffer(result[48:52], dtype=np.float32), [1])
        npt.assert_almost_equal(
            np.frombuffer(result[52:60], dtype=np.float32), [-0.03, -0.04]
        )

        assert int.from_bytes(result[60:64], sys.byteorder) == 3
        assert int.from_bytes(result[64:68], sys.byteorder) == 1
        npt.assert_equal(np.frombuffer(result[68:92], dtype=np.int64), [1, 3, 7])
        npt.assert_almost_equal(np.frombuffer(result[92:96], dtype=np.float32), [5.5])

        assert int.from_bytes(result[96:100], sys.byteorder) == 2
        assert int.from_bytes(result[100:104], sys.byteorder) == 1
        npt.assert_equal(np.frombuffer(result[104:120], dtype=np.int64), [1, 3])
        npt.assert_almost_equal(
            np.frombuffer(result[120:128], dtype=np.float32), [5.5, 6.89]
        )


@pytest.mark.parametrize("graph_with_sparse_features", [JsonDecoder], indirect=True)
def test_sanity_edge_sparse_features_index(graph_with_sparse_features):
    output = tempfile.TemporaryDirectory()
    data_name, decoder = graph_with_sparse_features
    convert.MultiWorkersConverter(
        graph_path=data_name,
        partition_count=1,
        output_dir=output.name,
        decoder=decoder(),
    ).convert()
    with open("{}/edge_features_{}_{}.index".format(output.name, 0, 0), "rb") as ei:
        expected_size = 152
        result = ei.read(expected_size + 1)
        assert len(result) == expected_size
        actual = []
        for i in range(19):
            actual.append(int.from_bytes(result[i * 8 : (i + 1) * 8], sys.byteorder))
        assert actual[0:8] == [0, 24, 50, 84, 104, 132, 180, 180]
        assert actual[8:16] == [206, 206, 246, 286, 314, 354, 354, 362]
        assert actual[16:19] == [362, 362, 394]


@pytest.mark.parametrize("graph_with_sparse_features", [JsonDecoder], indirect=True)
def test_sanity_edge_sparse_features_data(graph_with_sparse_features):
    output = tempfile.TemporaryDirectory()
    data_name, decoder = graph_with_sparse_features
    convert.MultiWorkersConverter(
        graph_path=data_name,
        partition_count=1,
        output_dir=output.name,
        decoder=decoder(),
    ).convert()
    with open("{}/edge_features_{}_{}.data".format(output.name, 0, 0), "rb") as efd:
        expected_size = 394  # last value in edge_features_index
        result = efd.read(expected_size + 1)
        assert len(result) == expected_size
        npt.assert_equal(np.frombuffer(result[0:24], dtype=np.uint64), [1, 2, 3])

        assert int.from_bytes(result[24:28], sys.byteorder) == 2
        assert int.from_bytes(result[28:32], sys.byteorder) == 1
        npt.assert_equal(np.frombuffer(result[32:48], dtype=np.int64), [7, 1])
        npt.assert_almost_equal(np.frombuffer(result[48:50], dtype=np.int8), [2, 1])

        assert int.from_bytes(result[50:54], sys.byteorder) == 3
        assert int.from_bytes(result[54:58], sys.byteorder) == 3
        npt.assert_equal(np.frombuffer(result[58:82], dtype=np.int64), [12, 5, 3])
        npt.assert_almost_equal(np.frombuffer(result[82:84], dtype=np.uint16), [65535])

        assert int.from_bytes(result[84:88], sys.byteorder) == 1
        assert int.from_bytes(result[88:92], sys.byteorder) == 1
        npt.assert_equal(np.frombuffer(result[92:100], dtype=np.int64), [9])
        npt.assert_almost_equal(
            np.frombuffer(result[100:104], dtype=np.uint32), [4294967295]
        )

        assert int.from_bytes(result[104:108], sys.byteorder) == 2
        assert int.from_bytes(result[108:112], sys.byteorder) == 1
        npt.assert_equal(np.frombuffer(result[112:128], dtype=np.int64), [7, 3])
        npt.assert_almost_equal(
            np.frombuffer(result[128:132], dtype=np.int16), [255, 16]
        )

        assert int.from_bytes(result[132:136], sys.byteorder) == 4
        assert int.from_bytes(result[136:140], sys.byteorder) == 2
        npt.assert_equal(np.frombuffer(result[140:172], dtype=np.int64), [5, 13, 7, 25])
        npt.assert_almost_equal(
            np.frombuffer(result[172:180], dtype=np.int32), [-1, 1024]
        )

        assert int.from_bytes(result[180:184], sys.byteorder) == 2
        assert int.from_bytes(result[184:188], sys.byteorder) == 1
        npt.assert_equal(np.frombuffer(result[188:204], dtype=np.int64), [5, 13])
        npt.assert_almost_equal(
            np.frombuffer(result[204:206], dtype=np.uint8), [10, 255]
        )

        assert int.from_bytes(result[206:210], sys.byteorder) == 2
        assert int.from_bytes(result[210:214], sys.byteorder) == 1
        npt.assert_equal(
            np.frombuffer(result[214:230], dtype=np.int64), [4294967296, 8294967296]
        )
        npt.assert_almost_equal(
            np.frombuffer(result[230:246], dtype=np.int64), [4294967296, 23]
        )

        assert int.from_bytes(result[246:250], sys.byteorder) == 2
        assert int.from_bytes(result[250:254], sys.byteorder) == 1
        npt.assert_equal(np.frombuffer(result[254:270], dtype=np.int64), [0, 5])
        npt.assert_almost_equal(
            np.frombuffer(result[270:286], dtype=np.double), [1.0, 2.13]
        )

        assert int.from_bytes(result[286:290], sys.byteorder) == 2
        assert int.from_bytes(result[290:294], sys.byteorder) == 1
        npt.assert_equal(np.frombuffer(result[294:310], dtype=np.int64), [1, -1])
        npt.assert_allclose(
            np.frombuffer(result[310:314], dtype=np.float16), [0.55, 0.33], rtol=0.001
        )

        assert int.from_bytes(result[314:318], sys.byteorder) == 2
        assert int.from_bytes(result[318:322], sys.byteorder) == 1
        npt.assert_equal(np.frombuffer(result[322:338], dtype=np.int64), [2, 3])
        npt.assert_almost_equal(
            np.frombuffer(result[338:354], dtype=np.uint64), [1, 8294967296]
        )

        npt.assert_allclose(np.frombuffer(result[354:362], dtype=np.float32), [3, 4])

        assert int.from_bytes(result[362:366], sys.byteorder) == 2
        assert int.from_bytes(result[366:370], sys.byteorder) == 1
        npt.assert_equal(np.frombuffer(result[370:386], dtype=np.int64), [1, 3])
        npt.assert_allclose(
            np.frombuffer(result[386:expected_size], dtype=np.float32), [5.5, 6.7]
        )


def graph_with_inverse_edge_type_order_json(folder):
    data = open(os.path.join(folder, "graph.json"), "w+")
    graph = [
        {
            "node_id": 9,
            "node_type": 0,
            "node_weight": 1,
            "edge": [
                {
                    "src_id": 9,
                    "dst_id": 1,
                    "edge_type": 3,
                    "weight": 1.0,
                },
                {
                    "src_id": 9,
                    "dst_id": 2,
                    "edge_type": 0,
                    "weight": 0.5,
                },
            ],
        },
    ]
    for el in graph:
        json.dump(el, data)
        data.write("\n")
    data.flush()
    data.close()

    return data.name


def graph_with_inverse_edge_type_order_tsv(folder):
    data = open(os.path.join(folder, "graph.tsv"), "w+")
    data.write("9\t0\t1\tf:0 1;f:-0.01 -0.02\t1,3,1.0|2,0,0.5\n")
    data.flush()
    data.close()
    return data.name


@pytest.fixture(scope="module")
def graph_with_inverse_edge_type_order(request):
    workdir = tempfile.TemporaryDirectory()
    if request.param == JsonDecoder:
        data_name = graph_with_inverse_edge_type_order_json(workdir.name)
    elif request.param == TsvDecoder:
        data_name = graph_with_inverse_edge_type_order_tsv(workdir.name)
    else:
        raise ValueError("Unsupported format.")

    yield data_name, request.param
    workdir.cleanup()


@pytest.mark.parametrize("graph_with_inverse_edge_type_order", param, indirect=True)
def test_edge_index_inverted_types(graph_with_inverse_edge_type_order):
    output = tempfile.TemporaryDirectory()
    data_name, decoder = graph_with_inverse_edge_type_order
    convert.MultiWorkersConverter(
        graph_path=data_name,
        partition_count=1,
        output_dir=output.name,
        decoder=decoder(),
    ).convert()
    with open("{}/edge_{}_{}.index".format(output.name, 0, 0), "rb") as ei:
        expected_size = 3 * 24  # 2 edges + last line as final close
        result = ei.read(expected_size + 100)
        assert len(result) == expected_size
        assert result[0:8] == (2).to_bytes(8, byteorder=sys.byteorder)
        assert result[8:16] == (0).to_bytes(8, byteorder=sys.byteorder)
        assert result[16:20] == (0).to_bytes(4, byteorder=sys.byteorder)
        assert result[20:24] == struct.pack("f", 0.5)

        assert result[24:32] == (1).to_bytes(8, byteorder=sys.byteorder)
        assert result[32:40] == (0).to_bytes(8, byteorder=sys.byteorder)
        assert result[40:44] == (3).to_bytes(4, byteorder=sys.byteorder)
        assert result[44:48] == struct.pack("f", 1.0)

        assert result[48:56] == (0).to_bytes(8, byteorder=sys.byteorder)
        assert result[56:64] == (0).to_bytes(8, byteorder=sys.byteorder)
        assert result[64:68] == (0).to_bytes(4, byteorder=sys.byteorder)
        assert result[68:72] == struct.pack("f", -1)


if __name__ == "__main__":
    sys.exit(
        pytest.main(
            [__file__, "--junitxml", os.environ["XML_OUTPUT_FILE"], *sys.argv[1:]]
        )
    )
