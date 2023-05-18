# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import json
import os
import sys
import tempfile
from pathlib import Path
import platform
from typing import List, Any

import numpy as np
import numpy.testing as npt
import pytest
import networkx as nx

import deepgnn.graph_engine.snark.client as client
from deepgnn.graph_engine.snark.decoders import JsonDecoder
import deepgnn.graph_engine.snark.convert as convert
import deepgnn.graph_engine.snark.server as server
import deepgnn.graph_engine.snark.dispatcher as dispatcher
import deepgnn.graph_engine.snark._lib as lib


def triangle_graph_json(folder):
    data = open(os.path.join(folder, "graph.json"), "w+")
    graph = [
        {
            "node_id": 9,
            "node_type": 0,
            "node_weight": 1,
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
    data.close()
    return data.name


@pytest.fixture(scope="module")
def triangle_graph_data():
    workdir = tempfile.TemporaryDirectory()
    data_name = triangle_graph_json(workdir.name)
    yield data_name
    workdir.cleanup()


def get_lib_name():
    lib_name = "libwrapper.so"
    if platform.system() == "Windows":
        lib_name = "wrapper.dll"
    elif platform.system() == "Darwin":
        lib_name = "libwrapper.dylib"
    return os.path.join(os.path.dirname(__file__), "..", lib_name)


def setup_module():
    lib._LIB_PATH = get_lib_name()


@pytest.fixture(scope="module")
def default_triangle_graph():
    output = tempfile.TemporaryDirectory()
    data_name = triangle_graph_data(output.name)
    convert.MultiWorkersConverter(
        graph_path=data_name,
        partition_count=1,
        output_dir=output.name,
    ).convert()
    yield output.name


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
    d = dispatcher.QueueDispatcher(Path(output.name), 2, Counter(), JsonDecoder())
    convert.MultiWorkersConverter(
        graph_path=data_name,
        partition_count=1,
        output_dir=output.name,
        decoder=JsonDecoder(),
        dispatcher=d,
    ).convert()

    yield output.name


def test_neighbor_sampling_graph_multiple_partitions(multi_partition_graph_data):
    g = client.MemoryGraph(
        multi_partition_graph_data, [(multi_partition_graph_data, 0)]
    )
    ns, ws, tp, ts = g.weighted_sample_neighbors(
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
    npt.assert_array_equal(ts, [[-1, -1], [-1, -1]])


def test_full_neighbor_graph_multiple_partitions(multi_partition_graph_data):
    g = client.MemoryGraph(
        multi_partition_graph_data,
        [(multi_partition_graph_data, 0), (multi_partition_graph_data, 1)],
    )
    node_ids, weights, types, counts = g.neighbors(
        nodes=np.array([9, 0, 5], dtype=np.int64), edge_types=1
    )
    npt.assert_array_equal(node_ids, [5, 9])
    npt.assert_array_equal(types, [1, 1])
    npt.assert_array_almost_equal(weights, [1, 0.7])
    npt.assert_array_equal(counts, [0, 1, 1])


def test_full_neighbor_graph_single_partition(multi_partition_graph_data):
    g = client.MemoryGraph(
        multi_partition_graph_data, [(multi_partition_graph_data, 1)]
    )
    node_ids, weights, types, counts = g.neighbors(
        nodes=np.array([9, 0, 5], dtype=np.int64), edge_types=1
    )
    npt.assert_array_equal(node_ids, [9])
    npt.assert_array_equal(types, [1])
    npt.assert_array_almost_equal(weights, [0.7])
    npt.assert_array_equal(counts, [0, 0, 1])


def test_full_neighbor_graph_handle_empty_list(multi_partition_graph_data):
    g = client.MemoryGraph(
        multi_partition_graph_data, [(multi_partition_graph_data, 0)]
    )
    node_ids, types, weights, counts = g.neighbors(
        nodes=np.array([9], dtype=np.int64), edge_types=1
    )
    npt.assert_array_equal(node_ids, [])
    npt.assert_array_equal(types, [])
    npt.assert_array_equal(weights, [])
    npt.assert_array_equal(counts, [0])


def test_neighbor_count_graph_single_partition(multi_partition_graph_data):
    g = client.MemoryGraph(
        multi_partition_graph_data, [(multi_partition_graph_data, 1)]
    )
    output_node_counts = g.neighbor_counts(
        nodes=np.array([9, 0, 5], dtype=np.int64), edge_types=1
    )

    npt.assert_array_equal(output_node_counts, [0, 0, 1])


def test_neighbor_count_graph_multiple_partitions(multi_partition_graph_data):
    g = client.MemoryGraph(
        multi_partition_graph_data,
        [(multi_partition_graph_data, 0), (multi_partition_graph_data, 1)],
    )
    output_node_counts = g.neighbor_counts(
        nodes=np.array([9, 0, 5], dtype=np.int64), edge_types=1
    )

    npt.assert_array_equal(output_node_counts, [0, 1, 1])


def test_neighbor_count_graph_handle_empty_list(multi_partition_graph_data):
    g = client.MemoryGraph(
        multi_partition_graph_data, [(multi_partition_graph_data, 0)]
    )
    output_node_counts = g.neighbor_counts(
        nodes=np.array([9], dtype=np.int64), edge_types=[100, 1]
    )

    npt.assert_array_equal(output_node_counts, [0])


def test_neighbor_count_graph_nonmatching_edge_type(multi_partition_graph_data):
    g = client.MemoryGraph(
        multi_partition_graph_data, [(multi_partition_graph_data, 0)]
    )
    output_node_counts = g.neighbor_counts(
        nodes=np.array([9], dtype=np.int64), edge_types=100
    )

    npt.assert_array_equal(output_node_counts, [0])


def test_neighbor_count_graph_nonexistent_node(multi_partition_graph_data):
    g = client.MemoryGraph(
        multi_partition_graph_data, [(multi_partition_graph_data, 0)]
    )
    output_node_counts = g.neighbor_counts(
        nodes=np.array([4], dtype=np.int64), edge_types=1
    )

    npt.assert_array_equal(output_node_counts, [0])


def test_neighbor_count_remote_client_single_partition(multi_partition_graph_data):
    s1 = server.Server(
        multi_partition_graph_data, [(multi_partition_graph_data, 0)], "localhost:12344"
    )
    s2 = server.Server(
        multi_partition_graph_data, [(multi_partition_graph_data, 1)], "localhost:12354"
    )

    cl = client.DistributedGraph(["localhost:12344", "localhost:12354"])
    output_node_counts = cl.neighbor_counts(
        nodes=np.array([9, 0, 5], dtype=np.int64), edge_types=1
    )

    print(output_node_counts)

    npt.assert_array_equal(output_node_counts, [0, 1, 1])
    s1.reset()
    s2.reset()


def test_neighbor_count_remote_client_multiple_partitions(multi_partition_graph_data):
    s1 = server.Server(
        multi_partition_graph_data, [(multi_partition_graph_data, 0)], "localhost:12345"
    )
    s2 = server.Server(
        multi_partition_graph_data, [(multi_partition_graph_data, 1)], "localhost:12355"
    )

    cl = client.DistributedGraph(["localhost:12345", "localhost:12355"])
    output_node_counts = cl.neighbor_counts(
        nodes=np.array([9, 0, 5], dtype=np.int64), edge_types=1
    )

    npt.assert_array_equal(output_node_counts, [0, 1, 1])
    s1.reset()
    s2.reset()


def test_neighbor_count_remote_client_handle_empty_list(multi_partition_graph_data):
    s1 = server.Server(
        multi_partition_graph_data, [(multi_partition_graph_data, 0)], "localhost:12346"
    )
    s2 = server.Server(
        multi_partition_graph_data, [(multi_partition_graph_data, 1)], "localhost:12356"
    )

    cl = client.DistributedGraph(["localhost:12346", "localhost:12356"])
    output_node_counts = cl.neighbor_counts(
        nodes=np.array([9], dtype=np.int64), edge_types=[100, 1]
    )

    npt.assert_array_equal(output_node_counts, [0])
    s1.reset()
    s2.reset()


def test_neighbor_count_remote_client_nonmatching_edge_type(multi_partition_graph_data):
    s1 = server.Server(
        multi_partition_graph_data, [(multi_partition_graph_data, 0)], "localhost:12347"
    )
    s2 = server.Server(
        multi_partition_graph_data, [(multi_partition_graph_data, 1)], "localhost:12357"
    )

    cl = client.DistributedGraph(["localhost:12347", "localhost:12357"])
    output_node_counts = cl.neighbor_counts(
        nodes=np.array([9], dtype=np.int64), edge_types=100
    )

    npt.assert_array_equal(output_node_counts, [0])
    s1.reset()
    s2.reset()


def test_neighbor_count_remote_client_nonexistent_node(multi_partition_graph_data):
    s1 = server.Server(
        multi_partition_graph_data, [(multi_partition_graph_data, 0)], "localhost:12358"
    )
    s2 = server.Server(
        multi_partition_graph_data, [(multi_partition_graph_data, 1)], "localhost:12368"
    )

    cl = client.DistributedGraph(["localhost:12358", "localhost:12368"])
    output_node_counts = cl.neighbor_counts(
        nodes=np.array([4], dtype=np.int64), edge_types=1
    )

    npt.assert_array_equal(output_node_counts, [0])
    s1.reset()
    s2.reset()


def test_neighbor_sampling_after_reset(multi_partition_graph_data):
    cl = client.MemoryGraph(
        multi_partition_graph_data, [(multi_partition_graph_data, 0)]
    )
    cl.reset()
    with pytest.raises(
        Exception, match="Failed to extract sampler neighbors with weights"
    ):
        cl.weighted_sample_neighbors(
            nodes=np.array([0, 1], dtype=np.int64), edge_types=1, count=2
        )


def test_uniform_neighbor_sampling_graph_multiple_partitions(
    multi_partition_graph_data,
):
    g = client.MemoryGraph(
        multi_partition_graph_data, [(multi_partition_graph_data, 0)]
    )
    for replacement in [True, False]:
        ns, tp, ts = g.uniform_sample_neighbors(
            replacement,
            nodes=np.array([0, 1], dtype=np.int64),
            edge_types=1,
            count=2,
            seed=3,
            default_node=13,
            default_type=-1,
        )
        if replacement:
            npt.assert_array_equal(ns, [[5, 13], [13, 13]])
            npt.assert_array_equal(tp, [[1, -1], [-1, -1]])
            npt.assert_array_equal(ts, [[-1, -1], [-1, -1]])
        else:
            npt.assert_array_equal(ns, [[5, 5], [13, 13]])
            npt.assert_array_equal(tp, [[1, 1], [-1, -1]])
            npt.assert_array_equal(ts, [[-1, -1], [-1, -1]])


def test_uniform_neighbor_sampling_after_reset(multi_partition_graph_data):
    cl = client.MemoryGraph(
        multi_partition_graph_data, [(multi_partition_graph_data, 0)]
    )
    cl.reset()
    with pytest.raises(
        Exception, match="Failed to sample neighbors with uniform distribution"
    ):
        cl.uniform_sample_neighbors(
            True, nodes=np.array([0, 1], dtype=np.int64), edge_types=1, count=2
        )


def test_remote_client_uniform_sampling_from_unsorted_types(multi_partition_graph_data):
    s1 = server.Server(
        multi_partition_graph_data, [(multi_partition_graph_data, 0)], "localhost:12349"
    )
    s2 = server.Server(
        multi_partition_graph_data, [(multi_partition_graph_data, 1)], "localhost:12359"
    )

    cl = client.DistributedGraph(["localhost:12349", "localhost:12359"])
    neighbors, types, _ = cl.uniform_sample_neighbors(
        False, nodes=np.array([1, 0], dtype=np.int64), edge_types=[0, 1], count=2
    )

    npt.assert_array_equal(neighbors, [[-1, -1], [5, 5]])
    npt.assert_array_equal(types, [[-1, -1], [1, 1]])
    s1.reset()
    s2.reset()


def test_remote_client_weighted_sampling_from_unsorted_types(
    multi_partition_graph_data,
):
    s1 = server.Server(
        multi_partition_graph_data, [(multi_partition_graph_data, 0)], "localhost:12359"
    )
    s2 = server.Server(
        multi_partition_graph_data, [(multi_partition_graph_data, 1)], "localhost:12369"
    )

    cl = client.DistributedGraph(["localhost:12359", "localhost:12369"])
    neighbors, weights, types, timestamps = cl.weighted_sample_neighbors(
        nodes=np.array([1, 0], dtype=np.int64), edge_types=[1, 0], count=2
    )

    npt.assert_array_equal(neighbors, [[-1, -1], [5, 5]])
    npt.assert_array_equal(types, [[-1, -1], [1, 1]])
    npt.assert_array_equal(weights, [[0, 0], [1.0, 1.0]])
    npt.assert_array_equal(timestamps, [[-1, -1], [-1, -1]])
    s1.reset()
    s2.reset()


def test_remote_client_weighted_sampling_with_missing_neighbors(
    multi_partition_graph_data,
):
    s1 = server.Server(
        multi_partition_graph_data, [(multi_partition_graph_data, 0)], "localhost:12378"
    )
    s2 = server.Server(
        multi_partition_graph_data, [(multi_partition_graph_data, 1)], "localhost:12379"
    )

    cl = client.DistributedGraph(["localhost:12378", "localhost:12379"])
    neighbors, weights, types, timestamps = cl.weighted_sample_neighbors(
        # Cover two cases: missing node(node_id=1) and missing neighbors(node_id=0)
        nodes=np.array([1, 0], dtype=np.int64),
        edge_types=[0],
        count=2,
        default_edge_type=-1,
        default_node=-1,
    )
    npt.assert_array_equal(neighbors, [[-1, -1], [-1, -1]])
    npt.assert_array_equal(types, [[-1, -1], [-1, -1]])
    npt.assert_array_equal(weights, [[0, 0], [0.0, 0.0]])
    npt.assert_array_equal(timestamps, [[-1, -1], [-1, -1]])
    s1.reset()
    s2.reset()


def test_remote_client_uniform_sampling_with_missing_neighbors(
    multi_partition_graph_data,
):
    s1 = server.Server(
        multi_partition_graph_data, [(multi_partition_graph_data, 0)], "localhost:12358"
    )
    s2 = server.Server(
        multi_partition_graph_data, [(multi_partition_graph_data, 1)], "localhost:12368"
    )
    cl = client.DistributedGraph(["localhost:12358", "localhost:12368"])

    for replacement in [True, False]:
        neighbors, types, timestamps = cl.uniform_sample_neighbors(
            without_replacement=replacement,
            # Cover two cases: missing node(node_id=1) and missing neighbors(node_id=0)
            nodes=np.array([1, 0], dtype=np.int64),
            edge_types=[0],
            count=2,
            default_type=-1,
            default_node=-1,
        )
        npt.assert_array_equal(neighbors, [[-1, -1], [-1, -1]])
        npt.assert_array_equal(types, [[-1, -1], [-1, -1]])
        npt.assert_array_equal(timestamps, [[-1, -1], [-1, -1]])

    s1.reset()
    s2.reset()


def karate_club_json(folder):
    data = open(os.path.join(folder, "graph.json"), "w+")
    raw = nx.karate_club_graph()
    graph: List[Any] = []
    for nx_node in raw.nodes():
        node_id = nx_node + 1
        node = {
            "node_id": node_id,
            "node_type": 0,
            "node_weight": 1,
            "edge": [],
        }
        for nb in raw.neighbors(nx_node):
            nb_index = nb + 1
            node["edge"].append(
                {"src_id": node_id, "dst_id": nb_index, "edge_type": 0, "weight": 1}
            )
        graph.append(node)

    for el in graph:
        json.dump(el, data)
        data.write("\n")
    data.flush()
    data.close()
    return data.name


@pytest.fixture(scope="module", params=["memory", "distributed"])
def karate_club_graph(request):
    with tempfile.TemporaryDirectory() as workdir:
        data_name = karate_club_json(workdir)
        d = dispatcher.QueueDispatcher(Path(workdir), 2, Counter(), JsonDecoder())
        convert.MultiWorkersConverter(
            graph_path=data_name,
            partition_count=2,
            output_dir=workdir,
            decoder=JsonDecoder(),
            dispatcher=d,
            skip_edge_sampler=True,
            skip_node_sampler=True,
        ).convert()

        if request.param == "memory":
            yield client.MemoryGraph(workdir, [(workdir, 0), (workdir, 1)])
        else:
            s1 = server.Server(workdir, [(workdir, 0)], "localhost:12359")
            s2 = server.Server(workdir, [(workdir, 1)], "localhost:12369")
            yield client.DistributedGraph(["localhost:12359", "localhost:12369"])
            s1.reset()
            s2.reset()


def test_karate_club_uniform_neighbor_sampling_different_result(
    karate_club_graph,
):
    for replacement in [True, False]:
        v1 = karate_club_graph.uniform_sample_neighbors(
            without_replacement=replacement,
            nodes=np.array([1, 7, 15], dtype=np.int64),
            edge_types=0,
            count=2,
        )[0]

        v2 = karate_club_graph.uniform_sample_neighbors(
            without_replacement=replacement,
            nodes=np.array([1, 7, 15], dtype=np.int64),
            edge_types=0,
            count=2,
        )[0]
        assert not np.array_equal(v1, v2)


def test_karate_club_weighted_neighbor_sampling_different_result(
    karate_club_graph,
):
    v1 = karate_club_graph.weighted_sample_neighbors(
        nodes=np.array([2, 4, 6], dtype=np.int64), edge_types=0, count=2
    )[0]

    v2 = karate_club_graph.weighted_sample_neighbors(
        nodes=np.array([2, 4, 6], dtype=np.int64), edge_types=0, count=2
    )[0]
    assert not np.array_equal(v1, v2)


def test_karate_club_ppr_sampling(
    karate_club_graph,
):
    nodes, weights, timestamps = karate_club_graph.ppr_neighbors(
        nodes=np.array([2, 4, 6, 8, 11, 13, 20, 22], dtype=np.int64),
        edge_types=0,
        count=5,
        alpha=0.1,
        eps=0.0001,
    )

    npt.assert_array_equal(
        nodes,
        [
            [4, 3, 34, 1, 2],
            [34, 3, 2, 1, 4],
            [17, 11, 7, 1, 6],
            [4, 3, 2, 1, 8],
            [7, 5, 6, 11, 1],
            [3, 2, 4, 13, 1],
            [33, 2, 34, 1, 20],
            [3, 34, 2, 22, 1],
        ],
    )
    npt.assert_array_almost_equal(
        weights,
        [
            [0.05211685, 0.06771754, 0.07165639, 0.11516844, 0.17243971],
            [0.06523439, 0.07632561, 0.0782424, 0.12137671, 0.15356852],
            [0.05838294, 0.06087079, 0.0871229, 0.14687803, 0.17239806],
            [0.06680953, 0.08266786, 0.08469853, 0.1229835, 0.13283907],
            [0.06148485, 0.06670617, 0.081143, 0.14703868, 0.156228],
            [0.06026748, 0.06429449, 0.08961273, 0.12202281, 0.15270987],
            [0.05358997, 0.08245811, 0.1060928, 0.11746016, 0.12044134],
            [0.05639789, 0.06037995, 0.10675042, 0.11910437, 0.1498548],
        ],
        verbose=True,
    )
    npt.assert_array_equal(timestamps, np.full((8, 5), -1, dtype=np.int64))


def test_karate_club_ppr_sampling_empty_nb_list(
    karate_club_graph,
):
    nodes, weights, _ = karate_club_graph.ppr_neighbors(
        nodes=np.array([1, 2, 6], dtype=np.int64),
        edge_types=12,
        count=2,
        alpha=0.2,
        eps=0.0001,
        default_node=-1,
        default_weight=0.0,
    )

    npt.assert_array_equal(nodes, [[1, -1], [2, -1], [6, -1]])
    npt.assert_array_almost_equal(
        weights,
        [
            [0.2, 0],
            [0.2, 0],
            [0.2, 0],
        ],
    )


def test_karate_club_ppr_sampling_missing_nodes(
    karate_club_graph,
):
    nodes, weights, _ = karate_club_graph.ppr_neighbors(
        nodes=np.array([35, 36, 37], dtype=np.int64),
        edge_types=0,
        count=2,
        alpha=0.1,
        eps=0.0001,
        default_node=-1,
        default_weight=0.0,
    )

    npt.assert_array_equal(nodes, [[35, -1], [36, -1], [37, -1]])
    npt.assert_array_almost_equal(
        weights,
        [
            [0.1, 0],
            [0.1, 0],
            [0.1, 0],
        ],
    )


def lastn_json(folder):
    data = open(os.path.join(folder, "graph.json"), "w+")
    graph: List[Any] = []
    node_count = 10
    nb_per_node = 10
    edge_type_count = 5
    edge_duration = 10
    for node_id in range(node_count):
        node = {
            "node_id": node_id,
            "node_type": 0,
            "node_weight": 1,
            "edge": [],
            "created_at": 0,
            "removed_at": -1,
        }
        for nb in range(nb_per_node):
            nb_id = node_id * nb_per_node + nb
            node["edge"].append(
                {
                    "src_id": node_id,
                    "dst_id": nb_id,
                    "edge_type": (nb_id % edge_type_count),
                    "weight": 1,
                    "created_at": (nb_id % edge_duration),
                    "removed_at": nb_id + edge_duration,
                }
            )
        graph.append(node)

    for el in graph:
        json.dump(el, data)
        data.write("\n")
    data.flush()
    data.close()
    return data.name


@pytest.fixture(scope="module", params=["memory", "distributed"])
def lastn_graph(request):
    with tempfile.TemporaryDirectory() as workdir:
        data_name = lastn_json(workdir)
        d = dispatcher.QueueDispatcher(
            Path(workdir), 2, Counter(), JsonDecoder(), watermark=0
        )
        convert.MultiWorkersConverter(
            graph_path=data_name,
            partition_count=2,
            output_dir=workdir,
            decoder=JsonDecoder(),
            dispatcher=d,
            skip_edge_sampler=True,
            skip_node_sampler=True,
            watermark=0,
            # debug=True,
        ).convert()

        if request.param == "memory":
            yield client.MemoryGraph(workdir, [(workdir, 0), (workdir, 1)])
        else:
            s1 = server.Server(workdir, [(workdir, 0)], "localhost:12379")
            s2 = server.Server(workdir, [(workdir, 1)], "localhost:12389")
            yield client.DistributedGraph(["localhost:12379", "localhost:12389"])
            s1.reset()
            s2.reset()


def test_lastn_neighbor_sampling(
    lastn_graph,
):
    nodes, weights, types, timestamps = lastn_graph.lastn_neighbors(
        nodes=np.array([1, 7, 15], dtype=np.int64),
        edge_types=[0, 1, 2, 3, 4, 5],
        count=2,
        timestamps=np.array([5, 7, 15], dtype=np.int64),
        default_weight=2,
        default_edge_type=42,
    )

    assert np.array_equal(nodes, [[14, 15], [76, 77], [-1, -1]])
    assert np.array_equal(weights, [[1, 1], [1, 1], [2, 2]])
    assert np.array_equal(types, [[4, 0], [1, 2], [42, 42]])
    assert np.array_equal(timestamps, [[4, 5], [6, 7], [-1, -1]])


def test_lastn_neighbor_sampling_edge_cases(
    lastn_graph,
):
    nodes, weights, types, timestamps = lastn_graph.lastn_neighbors(
        nodes=np.array([1, 7, 8], dtype=np.int64),
        edge_types=[0, 1, 2, 3, 4, 5],
        count=4,
        timestamps=np.array([1, -5, 1000], dtype=np.int64),
        default_weight=3,
        default_edge_type=13,
        default_timestamp=23,
    )

    assert np.array_equal(nodes, [[10, 11, -1, -1], [-1, -1, -1, -1], [-1, -1, -1, -1]])
    assert np.array_equal(weights, [[1, 1, 3, 3], [3, 3, 3, 3], [3, 3, 3, 3]])
    assert np.array_equal(types, [[0, 1, 13, 13], [13, 13, 13, 13], [13, 13, 13, 13]])
    assert np.array_equal(
        timestamps, [[0, 1, 23, 23], [23, 23, 23, 23], [23, 23, 23, 23]]
    )


if __name__ == "__main__":
    sys.exit(
        pytest.main(
            [__file__, "--junitxml", os.environ["XML_OUTPUT_FILE"], *sys.argv[1:]]
        )
    )
