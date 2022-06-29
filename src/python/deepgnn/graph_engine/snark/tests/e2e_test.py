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
import grpc
from grpc_health.v1 import health_pb2, health_pb2_grpc

import deepgnn.graph_engine.snark.client as client
from deepgnn.graph_engine.snark.decoders import JsonDecoder
import deepgnn.graph_engine.snark.server as server
import deepgnn.graph_engine.snark.convert as convert
import deepgnn.graph_engine.snark.dispatcher as dispatcher
import deepgnn.graph_engine.snark._lib as lib


def triangle_graph_json(folder):
    data = open(os.path.join(folder, "graph.json"), "w+")
    graph = [
        {
            "node_id": 9,
            "node_type": 0,
            "node_weight": 1,
            "neighbor": {"0": {"0": 0.5}, "1": {}},
            "uint64_feature": {"2": [13, 17]},
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
            "neighbor": {"0": {}, "1": {"5": 1}},
            "uint64_feature": {},
            "float_feature": {"0": [1], "1": [-0.03, -0.04]},
            "binary_feature": {"3": "abcd"},
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
            "neighbor": {"0": {}, "1": {"9": 0.7}},
            "float_feature": {"0": [1, 1], "1": [-0.05, -0.06]},
            "binary_feature": {},
            "uint8_feature": {"4": [5, 6, 7]},
            "int8_feature": {"5": [15, 16, 17]},
            "uint16_feature": {"6": [25, 26, 27]},
            "int16_feature": {"7": [35, 36, 37]},
            "uint32_feature": {"8": [45, 46, 47]},
            "int32_feature": {"9": [55, 56, 57]},
            "uint64_feature": {"10": [65, 66, 67]},
            "int64_feature": {"11": [75, 76, 77]},
            "double_feature": {"12": [85, 86, 87]},
            "float16_feature": {"13": [95, 96, 97]},
            "edge": [
                {
                    "src_id": 5,
                    "dst_id": 9,
                    "edge_type": 1,
                    "weight": 0.7,
                    "float_feature": {},
                    "binary_feature": {"2": "hello"},
                    "uint8_feature": {"4": [5, 6, 7]},
                    "int8_feature": {"5": [15, 16, 17]},
                    "uint16_feature": {"6": [25, 26, 27]},
                    "int16_feature": {"7": [35, 36, 37]},
                    "uint32_feature": {"8": [45, 46, 47]},
                    "int32_feature": {"9": [55, 56, 57]},
                    "uint64_feature": {"10": [65, 66, 67]},
                    "int64_feature": {"11": [75, 76, 77]},
                    "double_feature": {"12": [85, 86, 87]},
                    "float16_feature": {"13": [95, 96, 97]},
                }
            ],
        },
    ]
    for el in graph:
        json.dump(el, data)
        data.write("\n")
    data.flush()

    meta = open(os.path.join(folder, "meta.txt"), "w+")
    meta.write(
        '{"node_type_num": 3, "edge_type_num": 2, \
        "node_uint64_feature_num": 1, "node_float_feature_num": 2, \
        "node_binary_feature_num": 1, "edge_uint64_feature_num": 1, \
        "edge_float_feature_num": 1, "edge_binary_feature_num": 1}'
    )
    meta.flush()
    data.close()
    meta.close()
    return data.name, meta.name


def triangle_graph_tsv(folder):
    data = open(os.path.join(folder, "graph.tsv"), "w+")
    data.write("9\t0\t1\tf:0 1;f:-0.01 -0.02;u64:13 17\t0,0,0.5,u64:1 2 3\n")
    data.write("0\t1\t1\tf:1;f:-0.03 -0.04;;b:abcd\t5,1,1,;f:3 4\n")
    data.write(
        "5\t2\t1\tf:1 1;f:-0.05 -0.06;;;u8:5 6 7;i8:15 16 17;u16:25 26 27;i16:35 36 37;u32:45 46 47;i32:55 56 57;u64:65 66 67;i64:75 76 77;d:85 86 87;f16:95 96 97\t9,1,0.7,;;b:hello;;u8:5 6 7;i8:15 16 17;u16:25 26 27;i16:35 36 37;u32:45 46 47;i32:55 56 57;u64:65 66 67;i64:75 76 77;d:85 86 87;f16:95 96 97\n"
    )
    data.flush()

    meta = open(os.path.join(folder, "meta.txt"), "w+")
    meta.write(
        '{"node_type_num": 3, "edge_type_num": 2, \
        "node_uint64_feature_num": 1, "node_float_feature_num": 2, \
        "node_binary_feature_num": 1, "edge_uint64_feature_num": 1, \
        "edge_float_feature_num": 1, "edge_binary_feature_num": 1}'
    )
    meta.flush()

    data.close()
    meta.close()

    return data.name, meta.name


@pytest.fixture(scope="module")
def triangle_graph_data():
    workdir = tempfile.TemporaryDirectory()
    data_name, meta_name = triangle_graph_json(workdir.name)
    yield data_name, meta_name
    workdir.cleanup()


def get_lib_name():
    lib_name = "libwrapper.so"
    if platform.system() == "Windows":
        lib_name = "wrapper.dll"

    _SNARK_LIB_PATH_ENV_KEY = "SNARK_LIB_PATH"
    if _SNARK_LIB_PATH_ENV_KEY in os.environ:
        return os.environ[_SNARK_LIB_PATH_ENV_KEY]

    return os.path.join(os.path.dirname(__file__), "..", lib_name)


def setup_module(module):
    lib._LIB_PATH = get_lib_name()


@pytest.fixture(scope="module")
def default_triangle_graph():
    output = tempfile.TemporaryDirectory()
    data_name, meta_name = triangle_graph_json(output.name)
    convert.MultiWorkersConverter(
        graph_path=data_name,
        meta_path=meta_name,
        partition_count=1,
        output_dir=output.name,
        decoder_class=JsonDecoder,
    ).convert()
    yield output.name


@pytest.mark.parametrize(
    "storage_type",
    [client.PartitionStorageType.memory, client.PartitionStorageType.disk],
)
def test_node_features_graph_single_partition(default_triangle_graph, storage_type):
    cl = client.MemoryGraph(default_triangle_graph, [0], storage_type)
    v = cl.node_features(
        np.array([5], dtype=np.int64),
        features=np.array([[0, 2]], dtype=np.int32),
        dtype=np.float32,
    )
    assert v.shape == (1, 2)
    assert v[0][0] == 1.0
    assert v[0][1] == 1.0


@pytest.mark.parametrize(
    "storage_type",
    [client.PartitionStorageType.memory, client.PartitionStorageType.disk],
)
def test_multiple_node_features_graph_single_partition(
    default_triangle_graph, storage_type
):
    cl = client.MemoryGraph(default_triangle_graph, [0], storage_type)
    v = cl.node_features(
        np.array([0, 5], dtype=np.int64),
        features=np.array([[0, 2], [1, 2]], dtype=np.int32),
        dtype=np.float32,
    )
    assert v.shape == (2, 4)
    npt.assert_almost_equal(v, [[1, 0, -0.03, -0.04], [1, 1, -0.05, -0.06]])


@pytest.mark.parametrize(
    "storage_type",
    [client.PartitionStorageType.memory, client.PartitionStorageType.disk],
)
def test_node_sampling_graph_single_partition(default_triangle_graph, storage_type):
    graph = client.MemoryGraph(default_triangle_graph, [0], storage_type)
    ns = client.NodeSampler(graph, [0, 2])
    v, t = ns.sample(size=5, seed=2)
    npt.assert_array_equal(v, [9, 9, 5, 5, 5])
    npt.assert_array_equal(t, [0, 0, 2, 2, 2])


@pytest.mark.parametrize(
    "storage_type",
    [client.PartitionStorageType.memory, client.PartitionStorageType.disk],
)
def test_node_sampling_graph_default_seed_non_repeating(
    default_triangle_graph, storage_type
):
    graph = client.MemoryGraph(default_triangle_graph, [0], storage_type)
    ns = client.NodeSampler(graph, [0, 2])
    assert not np.array_equal(
        ns.sample(size=100, seed=1)[0], ns.sample(size=100, seed=2)[0]
    )


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
    data_name, meta_name = triangle_graph_json(output.name)
    d = dispatcher.QueueDispatcher(
        Path(output.name), 2, meta_name, Counter(), JsonDecoder
    )
    convert.MultiWorkersConverter(
        graph_path=data_name,
        meta_path=meta_name,
        partition_count=2,
        output_dir=output.name,
        decoder_class=JsonDecoder,
        dispatcher=d,
    ).convert()
    yield output.name


@pytest.mark.parametrize(
    "storage_type",
    [client.PartitionStorageType.memory, client.PartitionStorageType.disk],
)
def test_memory_graph_metadata(multi_partition_graph_data, storage_type):
    cl = client.MemoryGraph(multi_partition_graph_data, [0, 1], storage_type)
    assert cl.meta.node_count == 3
    assert cl.meta.edge_count == 3
    assert cl.meta.node_type_count == 3
    assert cl.meta.edge_type_count == 2


@pytest.mark.parametrize(
    "storage_type",
    [client.PartitionStorageType.memory, client.PartitionStorageType.disk],
)
def test_memory_graph_type_counts(multi_partition_graph_data, storage_type):
    cl = client.MemoryGraph(multi_partition_graph_data, [0, 1], storage_type)
    assert cl.get_node_type_count([0]) == 1
    assert cl.get_node_type_count([1]) == 1
    assert cl.get_node_type_count([1, 1]) == 1
    assert cl.get_node_type_count([0, 1]) == 2
    assert cl.get_node_type_count([1, 0]) == 2
    with pytest.raises(AssertionError):
        cl.get_node_type_count([])
    assert cl.get_edge_type_count([0]) == 1
    assert cl.get_edge_type_count([1]) == 2
    assert cl.get_edge_type_count([1, 1]) == 2
    assert cl.get_edge_type_count([0, 1]) == 3
    assert cl.get_edge_type_count([1, 0]) == 3
    with pytest.raises(AssertionError):
        cl.get_edge_type_count([])


@pytest.mark.parametrize(
    "storage_type",
    [client.PartitionStorageType.memory, client.PartitionStorageType.disk],
)
def test_multithreaded_calls(multi_partition_graph_data, storage_type):
    cl = client.MemoryGraph(multi_partition_graph_data, [0, 1], storage_type)
    num_calls = 100
    num_threads = 4

    def float_features():
        for _ in range(num_calls):
            v = cl.node_features(
                np.array([9, 0], dtype=np.int64),
                features=np.array([[1, 2]], dtype=np.int32),
                dtype=np.float32,
            )

            assert v.shape == (2, 2)
            npt.assert_array_almost_equal(v, [[-0.01, -0.02], [-0.03, -0.04]])

    def int_features():
        for _ in range(num_calls):
            v64 = cl.node_features(
                np.array([9], dtype=np.int64),
                features=np.array([[2, 2]], dtype=np.int32),
                dtype=np.uint64,
            )

            assert v64.shape == (1, 2)
            npt.assert_equal(v64, [[13, 17]])

    thread_list = []
    for _ in range(num_threads):
        thread_list.append(threading.Thread(target=float_features))
        thread_list.append(threading.Thread(target=int_features))
    [t.start() for t in thread_list]
    [t.join() for t in thread_list]


@pytest.mark.parametrize(
    "storage_type",
    [client.PartitionStorageType.memory, client.PartitionStorageType.disk],
)
def test_node_features_graph_multiple_partitions(
    multi_partition_graph_data, storage_type
):
    cl = client.MemoryGraph(multi_partition_graph_data, [0], storage_type)
    sampler = client.NodeSampler(cl, np.array([2]))
    nodes = sampler.sample(3, 42)
    npt.assert_equal(nodes, [5, 5, 5])


@pytest.mark.parametrize(
    "storage_type",
    [client.PartitionStorageType.memory, client.PartitionStorageType.disk],
)
def test_node_string_features_graph_multiple_partitions(
    multi_partition_graph_data, storage_type
):
    cl = client.MemoryGraph(multi_partition_graph_data, [0, 1], storage_type)
    v, d = cl.node_string_features([0], features=[0, 1], dtype=np.float32)
    npt.assert_equal(d, [[1, 2]])
    npt.assert_array_almost_equal(v, [1, -0.03, -0.04])


@pytest.mark.parametrize(
    "storage_type",
    [client.PartitionStorageType.memory, client.PartitionStorageType.disk],
)
def test_node_features_graph_multiple_partitions(
    multi_partition_graph_data, storage_type
):
    cl = client.MemoryGraph(multi_partition_graph_data, [0, 1], storage_type)
    v = cl.node_features(
        np.array([9, 0], dtype=np.int64),
        features=np.array([[1, 2]], dtype=np.int32),
        dtype=np.float32,
    )

    assert v.shape == (2, 2)
    npt.assert_array_almost_equal(v, [[-0.01, -0.02], [-0.03, -0.04]])

    v64 = cl.node_features(
        np.array([9], dtype=np.int64),
        features=np.array([[2, 2]], dtype=np.int32),
        dtype=np.uint64,
    )

    assert v64.shape == (1, 2)
    npt.assert_equal(v64, [[13, 17]])
    v_binary = cl.node_features(
        np.array([0], dtype=np.int64),
        features=np.array([[3, 4]], dtype=np.int32),
        dtype=np.uint8,
    )

    assert v_binary.shape == (1, 4)
    npt.assert_equal(v_binary, [list(map(int, bytes("abcd", "utf-8")))])

    v_missing = cl.node_features(
        np.array([-1, 9], dtype=np.int64),
        features=np.array([[2, 3]], dtype=np.int32),
        dtype=np.uint64,
    )

    assert v_missing.shape == (2, 3)
    npt.assert_equal(v_missing, [[0, 0, 0], [13, 17, 0]])


@pytest.mark.parametrize(
    "storage_type",
    [client.PartitionStorageType.memory, client.PartitionStorageType.disk],
)
def test_node_extra_features_graph_multiple_partitions(
    multi_partition_graph_data, storage_type
):
    cl = client.MemoryGraph(multi_partition_graph_data, [0, 1], storage_type)
    types = [
        np.uint8,
        np.int8,
        np.uint16,
        np.int16,
        np.uint32,
        np.int32,
        np.uint64,
        np.int64,
        np.double,
        np.float16,
    ]
    values = [5, 6, 7]
    feature_id = 4
    for tp in types:
        v = cl.node_features(
            np.array([5], dtype=np.int64),
            features=np.array([[feature_id, 3]], dtype=np.int32),
            dtype=tp,
        )

        assert v.shape == (1, 3)
        npt.assert_array_almost_equal(v, [values])
        values = list(map(lambda x: x + 10, values))
        feature_id += 1


@pytest.mark.parametrize(
    "storage_type",
    [client.PartitionStorageType.memory, client.PartitionStorageType.disk],
)
def test_edge_features_missing_feature_id(multi_partition_graph_data, storage_type):
    cl = client.MemoryGraph(multi_partition_graph_data, [0, 1], storage_type)
    v = cl.edge_features(
        np.array([0], dtype=np.int64),
        np.array([5], dtype=np.int64),
        np.array([1], dtype=np.int32),
        features=np.array([[2, 3]], dtype=np.int32),
        dtype=np.int32,
    )

    assert v.shape == (1, 3)
    npt.assert_equal(v, [[0, 0, 0]])


@pytest.mark.parametrize(
    "storage_type",
    [client.PartitionStorageType.memory, client.PartitionStorageType.disk],
)
def test_edge_extra_features_graph_multiple_partitions(
    multi_partition_graph_data, storage_type
):
    cl = client.MemoryGraph(multi_partition_graph_data, [0, 1], storage_type)
    types = [
        np.uint8,
        np.int8,
        np.uint16,
        np.int16,
        np.uint32,
        np.int32,
        np.uint64,
        np.int64,
        np.double,
        np.float16,
    ]
    values = [5, 6, 7]
    feature_id = 4
    for tp in types:
        v = cl.edge_features(
            np.array([5], dtype=np.int64),
            np.array([9], dtype=np.int64),
            np.array([1], dtype=np.int32),
            features=np.array([[feature_id, 3]], dtype=np.int32),
            dtype=tp,
        )

        assert v.shape == (1, 3)
        npt.assert_array_almost_equal(v, [values])
        values = list(map(lambda x: x + 10, values))
        feature_id += 1


def test_node_sampling_graph_multiple_partitions(multi_partition_graph_data):
    g = client.MemoryGraph(multi_partition_graph_data, [0, 1])
    ns = client.NodeSampler(g, [0, 2])
    v, t = ns.sample(size=5, seed=1)
    npt.assert_array_equal(v, [9, 9, 5, 5, 5])
    npt.assert_array_equal(t, [0, 0, 2, 2, 2])


def test_node_sampling_graph_with_only_missing_type(multi_partition_graph_data):
    g = client.MemoryGraph(multi_partition_graph_data, [0])
    with pytest.raises(AssertionError):
        client.NodeSampler(g, [4])


def test_node_sampling_graph_ok_with_extra_missing_type(multi_partition_graph_data):
    g = client.MemoryGraph(multi_partition_graph_data, [0, 1])
    ns = client.NodeSampler(g, [0, 4])
    v, t = ns.sample(size=3, seed=1)
    npt.assert_array_equal(v, [9, 9, 9])
    npt.assert_array_equal(t, [0, 0, 0])


def test_edge_sampling_non_repeating_defaults(multi_partition_graph_data):
    g = client.MemoryGraph(multi_partition_graph_data, [0, 1])
    es = client.EdgeSampler(g, [0, 1])
    v1 = es.sample(size=15, seed=1)
    v2 = es.sample(size=15, seed=2)
    assert not (
        np.array_equal(v1[0], v2[0])
        and np.array_equal(v1[1], v2[1])
        and np.array_equal(v1[2], v2[2])
    )


def test_edge_sampling_graph_multiple_partitions(multi_partition_graph_data):
    g = client.MemoryGraph(multi_partition_graph_data, [0, 1])
    es = client.EdgeSampler(g, [0, 1])
    s, d, t = es.sample(size=5, seed=2)
    npt.assert_array_equal(s, [9, 0, 0, 5, 5])
    npt.assert_array_equal(d, [0, 5, 5, 9, 9])
    npt.assert_array_equal(t, [0, 1, 1, 1, 1])


def test_edge_sampling_without_replacement_graph_multiple_partitions(
    multi_partition_graph_data,
):
    g = client.MemoryGraph(multi_partition_graph_data, [0, 1])
    es = client.EdgeSampler(g, [0, 1], "withoutreplacement")
    s, d, t = es.sample(size=5, seed=4)
    npt.assert_array_equal(s, [9, 0, 5, 9, 9])
    npt.assert_array_equal(d, [0, 5, 9, 5, 5])
    npt.assert_array_equal(t, [0, 1, 1, 1, 1])


def test_uniform_edge_sampling_graph_multiple_partitions(multi_partition_graph_data):
    g = client.MemoryGraph(multi_partition_graph_data, [0, 1])
    es = client.EdgeSampler(g, [0, 1], "uniform")
    s, d, t = es.sample(size=5, seed=4)
    npt.assert_array_equal(s, [9, 0, 5, 5, 5])
    npt.assert_array_equal(d, [0, 5, 9, 9, 9])
    npt.assert_array_equal(t, [0, 1, 1, 1, 1])


def test_edge_sampling_graph_single_partition(triangle_graph_data):
    output = tempfile.TemporaryDirectory()
    data_name, meta_name = triangle_graph_data
    convert.MultiWorkersConverter(
        graph_path=data_name,
        meta_path=meta_name,
        partition_count=1,
        output_dir=output.name,
        decoder_class=JsonDecoder,
    ).convert()

    g = client.MemoryGraph(output.name, [0])
    es = client.EdgeSampler(g, [0])
    s, d, t = es.sample(size=5, seed=3)
    npt.assert_array_equal(s, [9, 9, 9, 9, 9])
    npt.assert_array_equal(d, [0, 0, 0, 0, 0])
    npt.assert_array_equal(t, [0, 0, 0, 0, 0])


def test_edge_sampling_graph_single_partition_raises_empty_types(triangle_graph_data):
    output = tempfile.TemporaryDirectory()
    data_name, meta_name = triangle_graph_data
    convert.MultiWorkersConverter(
        graph_path=data_name,
        meta_path=meta_name,
        partition_count=1,
        output_dir=output.name,
        decoder_class=JsonDecoder,
    ).convert()

    g = client.MemoryGraph(output.name, [0, 1])
    with pytest.raises(AssertionError):
        client.EdgeSampler(g, [10])


def test_feature_extraction_after_reset(multi_partition_graph_data):
    cl = client.MemoryGraph(multi_partition_graph_data, [0])
    cl.reset()
    with pytest.raises(Exception, match="Failed to extract node features"):
        cl.node_features(
            np.array([9], dtype=np.int64),
            features=np.array([[1, 2]], dtype=np.int32),
            dtype=np.float32,
        )


def test_edge_sampler_creation(multi_partition_graph_data):
    cl = client.MemoryGraph(multi_partition_graph_data, [0])
    cl.reset()
    with pytest.raises(Exception, match="Failed to create edge sampler"):
        client.EdgeSampler(cl, [0])


def test_node_sampler_creation(multi_partition_graph_data):
    cl = client.MemoryGraph(multi_partition_graph_data, [0])
    cl.reset()
    with pytest.raises(Exception, match="Failed to create node sampler"):
        client.NodeSampler(cl, [0])


def test_edge_sampler_reset(multi_partition_graph_data):
    cl = client.MemoryGraph(multi_partition_graph_data, [0])
    es = client.EdgeSampler(cl, [0])
    es.reset()
    with pytest.raises(Exception, match="Failed to sample edges"):
        es.sample(5)


def test_node_sampler_reset(multi_partition_graph_data):
    cl = client.MemoryGraph(multi_partition_graph_data, [0])
    ns = client.NodeSampler(cl, [0])
    ns.reset()
    with pytest.raises(Exception, match="Failed to sample nodes"):
        ns.sample(5)


@pytest.mark.parametrize(
    "storage_type",
    [client.PartitionStorageType.memory, client.PartitionStorageType.disk],
)
def test_remote_client_node_features_single_server(
    multi_partition_graph_data, storage_type
):
    address = ["localhost:9999"]
    if platform.system() != "Darwin":
        address = [a + f"{int(storage_type)}" for a in address]
    s = server.Server(
        multi_partition_graph_data, [0], address[0], storage_type=storage_type
    )
    cl = client.DistributedGraph(address)
    v = cl.node_features(
        np.array([9, 0], dtype=np.int64),
        features=np.array([[0, 2], [1, 2]], dtype=np.int32),
        dtype=np.float32,
    )

    assert v.shape == (2, 4)
    npt.assert_array_almost_equal(v, [[0, 0, 0, 0], [1, 0, -0.03, -0.04]])
    s.reset()


@pytest.mark.parametrize(
    "storage_type",
    [client.PartitionStorageType.memory, client.PartitionStorageType.disk],
)
def test_distributed_graph_metadata(multi_partition_graph_data, storage_type):
    address = ["localhost:2234", "localhost:2235"]
    if platform.system() != "Darwin":
        address = [a + f"{int(storage_type)}" for a in address]

    s1 = server.Server(
        multi_partition_graph_data, [0], address[0], storage_type=storage_type
    )
    s2 = server.Server(
        multi_partition_graph_data, [1], address[1], storage_type=storage_type
    )

    cl = client.DistributedGraph(address)
    assert cl.meta.node_count == 3
    assert cl.meta.edge_count == 3
    assert cl.meta.node_type_count == 3
    assert cl.meta.edge_type_count == 2
    s1.reset()
    s2.reset()


@pytest.mark.parametrize(
    "storage_type",
    [client.PartitionStorageType.memory, client.PartitionStorageType.disk],
)
def test_distributed_graph_type_counts(multi_partition_graph_data, storage_type):
    address = ["localhost:12234", "localhost:12235"]
    if platform.system() != "Darwin":
        address = [a + f"{int(storage_type)}" for a in address]
    s1 = server.Server(
        multi_partition_graph_data, [0], address[0], storage_type=storage_type
    )
    s2 = server.Server(
        multi_partition_graph_data, [1], address[1], storage_type=storage_type
    )

    cl = client.DistributedGraph(address)
    assert cl.get_node_type_count([0]) == 1
    assert cl.get_node_type_count([1]) == 1
    assert cl.get_node_type_count([1, 1]) == 1
    assert cl.get_node_type_count([0, 1]) == 2
    assert cl.get_node_type_count([1, 0]) == 2
    with pytest.raises(AssertionError):
        cl.get_node_type_count([])
    assert cl.get_edge_type_count([0]) == 1
    assert cl.get_edge_type_count([1]) == 2
    assert cl.get_edge_type_count([1, 1]) == 2
    assert cl.get_edge_type_count([0, 1]) == 3
    assert cl.get_edge_type_count([1, 0]) == 3
    with pytest.raises(AssertionError):
        cl.get_edge_type_count([])

    s1.reset()
    s2.reset()


@pytest.mark.parametrize(
    "storage_type",
    [client.PartitionStorageType.memory, client.PartitionStorageType.disk],
)
def test_remote_client_node_features_multiple_servers(
    multi_partition_graph_data, storage_type
):
    address = ["localhost:1234", "localhost:1235"]
    if platform.system() != "Darwin":
        address = [a + f"{int(storage_type)}" for a in address]
    s1 = server.Server(
        multi_partition_graph_data, [0], address[0], storage_type=storage_type
    )
    s2 = server.Server(
        multi_partition_graph_data, [1], address[1], storage_type=storage_type
    )

    cl = client.DistributedGraph(address)
    v = cl.node_features(
        np.array([9, 0], dtype=np.int64),
        features=np.array([[1, 2]], dtype=np.int32),
        dtype=np.float32,
    )

    assert v.shape == (2, 2)
    npt.assert_array_almost_equal(v, [[-0.01, -0.02], [-0.03, -0.04]])
    s1.reset()
    s2.reset()


@pytest.mark.parametrize(
    "storage_type",
    [client.PartitionStorageType.memory, client.PartitionStorageType.disk],
)
def test_remote_client_node_string_features_multiple_servers(
    multi_partition_graph_data, storage_type
):
    address = ["localhost:1234", "localhost:1235"]
    if platform.system() != "Darwin":
        address = [a + f"{int(storage_type)}" for a in address]
    s1 = server.Server(
        multi_partition_graph_data, [0], address[0], storage_type=storage_type
    )
    s2 = server.Server(
        multi_partition_graph_data, [1], address[1], storage_type=storage_type
    )

    cl = client.DistributedGraph(address)
    v, d = cl.node_string_features([9, 0], features=[1], dtype=np.float32)

    npt.assert_array_equal(d, [[2], [2]])
    npt.assert_array_almost_equal(v, [-0.01, -0.02, -0.03, -0.04])
    s1.reset()
    s2.reset()


@pytest.mark.parametrize(
    "storage_type",
    [client.PartitionStorageType.memory, client.PartitionStorageType.disk],
)
def test_remote_client_node_extra_features(multi_partition_graph_data, storage_type):
    address = ["localhost:11234", "localhost:11235"]
    if platform.system() != "Darwin":
        address = [a + f"{int(storage_type)}" for a in address]
    s1 = server.Server(
        multi_partition_graph_data, [0], address[0], storage_type=storage_type
    )
    s2 = server.Server(
        multi_partition_graph_data, [1], address[1], storage_type=storage_type
    )

    cl = client.DistributedGraph(address)
    types = [
        np.uint8,
        np.int8,
        np.uint16,
        np.int16,
        np.uint32,
        np.int32,
        np.uint64,
        np.int64,
        np.double,
        np.float16,
    ]
    values = [5, 6, 7]
    feature_id = 4
    for tp in types:
        v = cl.node_features(
            np.array([5], dtype=np.int64),
            features=np.array([[feature_id, 3]], dtype=np.int32),
            dtype=tp,
        )

        assert v.shape == (1, 3)
        npt.assert_array_almost_equal(v, [values])
        values = list(map(lambda x: x + 10, values))
        feature_id += 1

    s1.reset()
    s2.reset()


@pytest.mark.parametrize(
    "storage_type",
    [client.PartitionStorageType.memory, client.PartitionStorageType.disk],
)
def test_remote_client_edge_extra_features_graph_multiple_partitions(
    multi_partition_graph_data, storage_type
):
    address = ["localhost:9234", "localhost:9235"]
    if platform.system() != "Darwin":
        address = [a + f"{int(storage_type)}" for a in address]
    s1 = server.Server(
        multi_partition_graph_data, [0], address[0], storage_type=storage_type
    )
    s2 = server.Server(
        multi_partition_graph_data, [1], address[1], storage_type=storage_type
    )

    cl = client.DistributedGraph(address)
    types = [
        np.uint8,
        np.int8,
        np.uint16,
        np.int16,
        np.uint32,
        np.int32,
        np.uint64,
        np.int64,
        np.double,
        np.float16,
    ]
    values = [5, 6, 7]
    feature_id = 4
    for tp in types:
        v = cl.edge_features(
            np.array([5], dtype=np.int64),
            np.array([9], dtype=np.int64),
            np.array([1], dtype=np.int32),
            features=np.array([[feature_id, 3]], dtype=np.int32),
            dtype=tp,
        )

        assert v.shape == (1, 3)
        npt.assert_array_almost_equal(v, [values])
        values = list(map(lambda x: x + 10, values))
        feature_id += 1
    s1.reset()
    s2.reset()


@pytest.mark.parametrize(
    "storage_type",
    [client.PartitionStorageType.memory, client.PartitionStorageType.disk],
)
def test_remote_client_missing_edge_features_graph_multiple_partitions(
    multi_partition_graph_data, storage_type
):
    address = ["localhost:9234", "localhost:9235"]
    if platform.system() != "Darwin":
        address = [a + f"{int(storage_type)}" for a in address]
    s1 = server.Server(
        multi_partition_graph_data, [0], address[0], storage_type=storage_type
    )
    s2 = server.Server(
        multi_partition_graph_data, [1], address[1], storage_type=storage_type
    )

    cl = client.DistributedGraph(address)
    v = cl.edge_features(
        np.array([5, -1], dtype=np.int64),
        np.array([9, 0], dtype=np.int64),
        np.array([1, 1], dtype=np.int32),
        features=np.array([[4, 3], [5, 2]], dtype=np.int32),
        dtype=np.uint8,
    )

    npt.assert_array_equal(v, [[5, 6, 7, 15, 16], [0, 0, 0, 0, 0]])
    s1.reset()
    s2.reset()


@pytest.mark.parametrize(
    "storage_type",
    [client.PartitionStorageType.memory, client.PartitionStorageType.disk],
)
def test_remote_client_missing_edge_string_features_graph_multiple_partitions(
    multi_partition_graph_data, storage_type
):
    address = ["localhost:9236", "localhost:9237"]
    if platform.system() != "Darwin":
        address = [a + f"{int(storage_type)}" for a in address]
    s1 = server.Server(
        multi_partition_graph_data, [0], address[0], storage_type=storage_type
    )
    s2 = server.Server(
        multi_partition_graph_data, [1], address[1], storage_type=storage_type
    )

    cl = client.DistributedGraph(address)
    v, d = cl.edge_string_features(
        np.array([5, -1], dtype=np.int64),
        np.array([9, 0], dtype=np.int64),
        np.array([1, 1], dtype=np.int32),
        features=np.array([4, 5], dtype=np.int8),
        dtype=np.uint8,
    )

    npt.assert_equal(d, [[3, 3], [0, 0]])
    npt.assert_equal(v, [5, 6, 7, 15, 16, 17])
    s1.reset()
    s2.reset()


@pytest.mark.parametrize(
    "storage_type",
    [client.PartitionStorageType.memory, client.PartitionStorageType.disk],
)
def test_remote_client_node_features_multiple_servers_same_data_tst(
    multi_partition_graph_data, storage_type
):
    address = ["localhost:1236", "localhost:1237"]
    if platform.system() != "Darwin":
        address = [a + f"{int(storage_type)}" for a in address]
    s1 = server.Server(
        multi_partition_graph_data, [0, 1], address[0], storage_type=storage_type
    )
    s2 = server.Server(
        multi_partition_graph_data, [1, 0], address[1], storage_type=storage_type
    )
    cl = client.DistributedGraph(address)
    v = cl.node_features(
        np.array([9, 0], dtype=np.int64),
        features=np.array([[1, 2]], dtype=np.int32),
        dtype=np.float32,
    )

    assert v.shape == (2, 2)
    npt.assert_array_almost_equal(v, [[-0.01, -0.02], [-0.03, -0.04]])
    s1.reset()
    s2.reset()


@pytest.mark.parametrize(
    "storage_type",
    [client.PartitionStorageType.memory, client.PartitionStorageType.disk],
)
def test_remote_client_node_features_multiple_servers_connection_loss(
    multi_partition_graph_data, storage_type
):
    address = ["localhost:1238", "localhost:1239"]
    if platform.system() != "Darwin":
        address = [a + f"{int(storage_type)}" for a in address]
    s1 = server.Server(
        multi_partition_graph_data, [0], hostname=address[0], storage_type=storage_type
    )
    s2 = server.Server(
        multi_partition_graph_data, [1], hostname=address[1], storage_type=storage_type
    )

    cl = client.DistributedGraph(address)

    before = cl.node_features(
        np.array([9, 0], dtype=np.int64),
        features=np.array([[1, 2]], dtype=np.int32),
        dtype=np.float32,
    )

    assert before.shape == (2, 2)
    npt.assert_array_almost_equal(before, [[-0.01, -0.02], [-0.03, -0.04]])

    s2.reset()
    with pytest.raises(Exception, match="Failed to extract node features"):
        cl.node_features(
            np.array([9, 0], dtype=np.int64),
            features=np.array([[1, 2]], dtype=np.int32),
            dtype=np.float32,
        )
    s1.reset()


@pytest.mark.parametrize(
    "storage_type",
    [client.PartitionStorageType.memory, client.PartitionStorageType.disk],
)
def test_node_sampling_distributed_graph_multiple_partitions(
    multi_partition_graph_data, storage_type
):
    address = ["localhost:1240", "localhost:1241"]
    if platform.system() != "Darwin":
        address = [a + f"{int(storage_type)}" for a in address]
    s1 = server.Server(multi_partition_graph_data, [0], hostname=address[0])
    s2 = server.Server(multi_partition_graph_data, [1], hostname=address[1])

    cl = client.DistributedGraph(address)

    ns = client.NodeSampler(cl, [0, 2])
    v, t = ns.sample(size=5, seed=1)
    npt.assert_array_equal(v, [9, 5, 5, 5, 5])
    npt.assert_array_equal(t, [0, 2, 2, 2, 2])
    s2.reset()
    s1.reset()


@pytest.mark.parametrize(
    "storage_type",
    [client.PartitionStorageType.memory, client.PartitionStorageType.disk],
)
def test_node_sampling_distributed_graph_multiple_partitions_raises_empty_types(
    multi_partition_graph_data, storage_type
):
    address = ["localhost:12409", "localhost:12419"]
    if platform.system() != "Darwin":
        address = [a + f"{int(storage_type)}" for a in address]
    s1 = server.Server(
        multi_partition_graph_data, [0], hostname=address[0], storage_type=storage_type
    )
    s2 = server.Server(
        multi_partition_graph_data, [1], hostname=address[1], storage_type=storage_type
    )

    cl = client.DistributedGraph(address)
    with pytest.raises(AssertionError):
        client.NodeSampler(cl, [10])
    s1.reset()
    s2.reset()


def test_edge_sampling_distributed_graph_multiple_partitions(
    multi_partition_graph_data,
):
    _ = server.Server(multi_partition_graph_data, [0], hostname="localhost:1243")
    _ = server.Server(multi_partition_graph_data, [1], hostname="localhost:1244")

    cl = client.DistributedGraph(["localhost:1243", "localhost:1244"])
    es = client.EdgeSampler(cl, [0, 1])
    s, d, t = es.sample(size=5, seed=2)
    npt.assert_array_equal(s, [0, 0, 5, 5, 5])
    npt.assert_array_equal(d, [5, 5, 9, 9, 9])
    npt.assert_array_equal(t, [1, 1, 1, 1, 1])


@pytest.mark.parametrize(
    "storage_type",
    [client.PartitionStorageType.memory, client.PartitionStorageType.disk],
)
def test_edge_sampling_distributed_graph_multiple_partitions_raises_empty_types(
    multi_partition_graph_data, storage_type
):
    address = ["localhost:12439", "localhost:12449"]
    if platform.system() != "Darwin":
        address = [a + f"{int(storage_type)}" for a in address]
    s1 = server.Server(
        multi_partition_graph_data, [0], hostname=address[0], storage_type=storage_type
    )
    s2 = server.Server(
        multi_partition_graph_data, [1], hostname=address[1], storage_type=storage_type
    )

    cl = client.DistributedGraph(address)
    with pytest.raises(AssertionError):
        client.EdgeSampler(cl, [10])
    s1.reset()
    s2.reset()


@pytest.mark.parametrize(
    "storage_type",
    [client.PartitionStorageType.memory, client.PartitionStorageType.disk],
)
def test_node_sampling_distributed_graph_multiple_partitions_server_down(
    multi_partition_graph_data, storage_type
):
    address = ["localhost:1250", "localhost:1251"]
    if platform.system() != "Darwin":
        address = [a + f"{int(storage_type)}" for a in address]
    s1 = server.Server(
        multi_partition_graph_data, [0], hostname=address[0], storage_type=storage_type
    )
    s2 = server.Server(
        multi_partition_graph_data, [1], hostname=address[1], storage_type=storage_type
    )

    cl = client.DistributedGraph(address)

    ns = client.NodeSampler(cl, [0, 2])

    s2.reset()
    with pytest.raises(Exception, match="Failed to sample nodes"):
        ns.sample(size=5, seed=1)
    s1.reset()


@pytest.mark.parametrize(
    "storage_type",
    [client.PartitionStorageType.memory, client.PartitionStorageType.disk],
)
def test_edge_sampling_distributed_graph_multiple_partitions_server_down(
    multi_partition_graph_data, storage_type
):
    address = ["localhost:1253", "localhost:1254"]
    if platform.system() != "Darwin":
        address = [a + f"{int(storage_type)}" for a in address]
    s1 = server.Server(
        multi_partition_graph_data, [0], hostname=address[0], storage_type=storage_type
    )
    s2 = server.Server(
        multi_partition_graph_data, [1], hostname=address[1], storage_type=storage_type
    )

    cl = client.DistributedGraph(address)
    es = client.EdgeSampler(cl, [0, 1])

    s2.reset()
    with pytest.raises(Exception, match="Failed to sample edges"):
        es.sample(size=5, seed=2)
    s1.reset()


class _TrainingWorker:
    def __init__(self, addresses: List[str], path: str, num_processes: int, dir: str):
        self.addresses = addresses
        self.path = path
        self.num_processes = num_processes
        self.dir = dir

    def __call__(self):
        os.environ[lib._SNARK_LIB_PATH_ENV_KEY] = self.path

        cl = client.DistributedGraph(self.addresses)
        cl.node_features(
            np.array([9, 0], dtype=np.int64),
            features=np.array([[1, 2]], dtype=np.int32),
            dtype=np.float32,
        )
        pid = os.getpid()
        with open(os.path.join(self.dir, f"{pid}.lock"), "w+") as tmp:
            tmp.write("done")

        while True:
            _ = cl.node_features(
                np.array([9, 0], dtype=np.int64),
                features=np.array([[1, 2]], dtype=np.int32),
                dtype=np.float32,
            )
            time.sleep(0.1)


@pytest.mark.parametrize(
    "storage_type",
    [client.PartitionStorageType.memory, client.PartitionStorageType.disk],
)
def test_servers_stay_alive_on_client_disconnects(
    multi_partition_graph_data, storage_type
):
    addresses = ["localhost:1255", "localhost:1256"]
    if platform.system() != "Darwin":
        addresses = [a + f"{int(storage_type)}" for a in addresses]
    s1 = server.Server(
        multi_partition_graph_data,
        [0],
        hostname=addresses[0],
        storage_type=storage_type,
    )
    s2 = server.Server(
        multi_partition_graph_data,
        [1],
        hostname=addresses[1],
        storage_type=storage_type,
    )
    trainers: List[Tuple(mp.Event, mp.Process)] = []
    num_processes = 10

    with tempfile.TemporaryDirectory(suffix="snark_") as workdir:
        for _ in range(num_processes):
            # Spawn child processes instead of forking to imitate distributed training
            p = mp.get_context("spawn").Process(
                target=_TrainingWorker(
                    addresses, get_lib_name(), num_processes, workdir
                )
            )
            p.start()
            trainers.append(p)
        timeout_count = 0
        while len(os.listdir(workdir)) < num_processes:
            time.sleep(1)
            timeout_count += 1
            assert timeout_count < 10

    for trainer in trainers:
        trainer.terminate()

    # final check: a client can connect to the servers after random terminations above.
    cl = client.DistributedGraph(addresses)
    values = cl.node_features(
        np.array([9, 0], dtype=np.int64),
        features=np.array([[1, 2]], dtype=np.int32),
        dtype=np.float32,
    )
    npt.assert_array_almost_equal(values, [[-0.01, -0.02], [-0.03, -0.04]])
    s2.reset()
    s1.reset()


@pytest.fixture(scope="module")
def sampling_graph_data():
    workdir = tempfile.TemporaryDirectory()
    data = open(os.path.join(workdir.name, "graph.json"), "w+")
    graph = []
    num_nodes = 10
    num_types = 3
    for node_id in range(num_nodes):
        graph.append(
            {
                "node_id": node_id,
                "node_type": (node_id % num_types),
                "node_weight": 1,
                "neighbor": {},
                "uint64_feature": None,
                "float_feature": None,
                "binary_feature": None,
                "edge": [],
            }
        )

    for el in graph:
        json.dump(el, data)
        data.write("\n")
    data.flush()

    meta = open(os.path.join(workdir.name, "meta.txt"), "w+")
    meta.write('{"node_type_num": ')
    meta.write(str(num_types))
    meta.write(
        ', "edge_type_num": 0, \
        "node_uint64_feature_num": 0, "node_float_feature_num": 0, \
        "node_binary_feature_num": 0, "edge_uint64_feature_num": 0, \
        "edge_float_feature_num": 0, "edge_binary_feature_num": 0}'
    )
    meta.flush()
    yield data.name, meta.name

    data.close()
    meta.close()
    workdir.cleanup()


@pytest.fixture(scope="module")
def default_node_sampling_graph(sampling_graph_data):
    output = tempfile.TemporaryDirectory()
    data_name, meta_name = sampling_graph_data
    convert.MultiWorkersConverter(
        graph_path=data_name,
        meta_path=meta_name,
        partition_count=1,
        output_dir=output.name,
        decoder_class=JsonDecoder,
    ).convert()

    yield output.name


@pytest.mark.parametrize(
    "input_types,expected_nodes,expected_types",
    [
        ([0], [0, 3, 6, 9], [0] * 4),
        ([0, 1], [0, 1, 3, 4, 6, 7, 9], [0] * 4 + [1] * 3),
        ([0, 1, 2], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0] * 4 + [1] * 3 + [2] * 3),
    ],
)
def test_node_sampling_graph_single_partition_all_nodes_withoutreplacement(
    default_node_sampling_graph, input_types, expected_nodes, expected_types
):
    graph = client.MemoryGraph(default_node_sampling_graph, [0])
    ns = client.NodeSampler(graph, input_types, "withoutreplacement")
    v, t = ns.sample(size=len(expected_nodes), seed=2)
    v.sort()
    t.sort()
    npt.assert_array_equal(v, expected_nodes)
    npt.assert_array_equal(t, expected_types)


def no_features_graph_json(folder):
    data = open(os.path.join(folder, "graph.json"), "w+")
    graph = [
        {
            "node_id": 9,
            "node_type": 0,
            "node_weight": 1,
            "neighbor": {"0": {"0": 0.5}, "1": {}},
            "uint64_feature": {},
            "float_feature": {},
            "binary_feature": {},
            "edge": [
                {
                    "src_id": 9,
                    "dst_id": 0,
                    "edge_type": 0,
                    "weight": 0.5,
                    "uint64_feature": {},
                    "float_feature": {},
                    "binary_feature": {},
                }
            ],
        },
        {
            "node_id": 0,
            "node_type": 1,
            "node_weight": 1,
            "neighbor": {"0": {}, "1": {"5": 1}},
            "uint64_feature": {},
            "float_feature": {},
            "binary_feature": {},
            "edge": [
                {
                    "src_id": 0,
                    "dst_id": 5,
                    "edge_type": 1,
                    "weight": 1,
                    "uint64_feature": {},
                    "float_feature": {},
                    "binary_feature": {},
                }
            ],
        },
    ]
    for el in graph:
        json.dump(el, data)
        data.write("\n")
    data.flush()

    meta = open(os.path.join(folder, "meta.txt"), "w+")
    meta.write(
        '{"node_type_num": 2, "edge_type_num": 2, \
        "node_uint64_feature_num": 0, "node_float_feature_num": 0, \
        "node_binary_feature_num": 0, "edge_uint64_feature_num": 0, \
        "edge_float_feature_num": 0, "edge_binary_feature_num": 0}'
    )
    meta.flush()
    data.close()
    meta.close()
    return data.name, meta.name


@pytest.fixture(scope="module")
def no_features_graph():
    output = tempfile.TemporaryDirectory()
    data_name, meta_name = no_features_graph_json(output.name)
    d = dispatcher.QueueDispatcher(
        Path(output.name), 2, meta_name, Counter(), JsonDecoder
    )

    convert.MultiWorkersConverter(
        graph_path=data_name,
        meta_path=meta_name,
        partition_count=2,
        output_dir=output.name,
        decoder_class=JsonDecoder,
        dispatcher=d,
        skip_edge_sampler=True,
        skip_node_sampler=True,
    ).convert()

    yield output.name
    output.cleanup()


@pytest.fixture(scope="module")
def in_memory_no_features_graph(no_features_graph):
    return client.MemoryGraph(no_features_graph, [0])


@pytest.fixture(scope="module")
def multi_server_no_features_graph(no_features_graph):
    s1 = server.Server(no_features_graph, [0], hostname="localhost:1255")
    s2 = server.Server(no_features_graph, [1], hostname="localhost:1256")

    yield client.DistributedGraph(["localhost:1255", "localhost:1256"])

    s1.reset()
    s2.reset()


def test_single_graph_partition_without_features(in_memory_no_features_graph):
    cl = in_memory_no_features_graph
    v = cl.node_features(
        np.array([9], dtype=np.int64),
        features=np.array([[0, 2]], dtype=np.int32),
        dtype=np.float32,
    )
    npt.assert_equal(v, np.zeros((1, 2), dtype=np.float32))

    ev = cl.edge_features(
        np.array([9], dtype=np.int64),
        np.array([0], dtype=np.int64),
        np.array([0], dtype=np.int32),
        features=np.array([[0, 3]], dtype=np.int32),
        dtype=np.float32,
    )

    npt.assert_equal(ev, np.zeros((1, 3), dtype=np.float32))


def test_multi_graph_partition_without_features(multi_server_no_features_graph):
    cl = multi_server_no_features_graph
    v = cl.node_features(
        np.array([9, 0], dtype=np.int64),
        features=np.array([[0, 3]], dtype=np.int32),
        dtype=np.float32,
    )
    npt.assert_equal(v, np.zeros((2, 3), dtype=np.float32))

    ev = cl.edge_features(
        np.array([9, 0], dtype=np.int64),
        np.array([0, 5], dtype=np.int64),
        np.array([0, 1], dtype=np.int32),
        features=np.array([[0, 3]], dtype=np.int32),
        dtype=np.float32,
    )

    npt.assert_equal(ev, np.zeros((2, 3), dtype=np.float32))


def test_node_types_in_memory(in_memory_no_features_graph):
    cl = in_memory_no_features_graph
    v = cl.node_types(np.array([9, 0, 42], dtype=np.int64), default_type=-1)
    npt.assert_equal(v, np.array([-1, 1, -1], dtype=np.int32))


def test_node_types_distributed(multi_server_no_features_graph):
    cl = multi_server_no_features_graph
    v = cl.node_types(np.array([42, 0, 9], dtype=np.int64), default_type=-1)
    npt.assert_equal(v, np.array([-1, 1, 0], dtype=np.int32))


def test_health_check(no_features_graph):
    s = server.Server(no_features_graph, [0], hostname="localhost:1257")
    channel = grpc.insecure_channel("localhost:1257")
    stub = health_pb2_grpc.HealthStub(channel)
    response = stub.Check(health_pb2.HealthCheckRequest(service=""))
    assert str(response) == "status: SERVING\n"
    s.reset()


if __name__ == "__main__":
    sys.exit(
        pytest.main(
            [__file__, "--junitxml", os.environ["XML_OUTPUT_FILE"], *sys.argv[1:]]
        )
    )
