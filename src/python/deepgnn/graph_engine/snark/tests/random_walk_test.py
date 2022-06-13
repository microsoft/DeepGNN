# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import json
import os
import sys
import tempfile
from pathlib import Path
import platform
from typing import List, Any
import random

import numpy as np
import numpy.testing as npt
import pytest
import networkx as nx

import deepgnn.graph_engine.snark.client as client
from deepgnn.graph_engine.snark.decoders import json_node_to_linear
import deepgnn.graph_engine.snark.server as server
import deepgnn.graph_engine.snark.convert as convert
import deepgnn.graph_engine.snark.dispatcher as dispatcher
import deepgnn.graph_engine.snark._lib as lib


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
            "neighbor": {"0": {}},
            "edge": [],
        }
        for nb in raw.neighbors(nx_node):
            nb_index = nb + 1
            node["neighbor"]["0"][str(nb_index)] = 1
            node["edge"].append(
                {"src_id": node_id, "dst_id": nb_index, "edge_type": 0, "weight": 1}
            )
        graph.append(node)

    for el in graph:
        data.write(json_node_to_linear(el))
    data.flush()

    meta = open(os.path.join(folder, "meta.txt"), "w+")
    meta.write(
        '{"node_type_num": 1, "edge_type_num": 1, \
        "node_uint64_feature_num": 0, "node_float_feature_num": 0, \
        "node_binary_feature_num": 0, "edge_uint64_feature_num": 0, \
        "edge_float_feature_num": 0, "edge_binary_feature_num": 0}'
    )
    meta.flush()
    data.close()
    meta.close()
    return data.name, meta.name


def get_lib_name():
    lib_name = "libwrapper.so"
    if platform.system() == "Windows":
        lib_name = "wrapper.dll"
    return os.path.join(os.path.dirname(__file__), "..", lib_name)


def setup_module(module):
    lib._LIB_PATH = get_lib_name()


# We'll use this class for deterministic partitioning
class Counter:
    def __init__(self):
        self.count = 0

    def __call__(self, x):
        self.count += 1
        return self.count % 2


@pytest.fixture(scope="module")
def binary_karate_club_data():
    with tempfile.TemporaryDirectory() as workdir:
        data_name, meta_name = karate_club_json(workdir)
        d = dispatcher.QueueDispatcher(
            Path(workdir), 2, meta_name, convert.output, Counter()
        )

        convert.MultiWorkersConverter(
            graph_path=data_name,
            meta_path=meta_name,
            partition_count=2,
            output_dir=workdir,
            dispatcher=d,
        ).convert()
        yield workdir


def test_karate_club_random_walk_memory(binary_karate_club_data):
    cl = client.MemoryGraph(binary_karate_club_data, [0, 1])
    walks = cl.random_walk(
        node_ids=np.array([1, 7, 15], dtype=np.int64),
        edge_types=0,
        walk_len=3,
        p=2,
        q=0.5,
        seed=1,
    )

    npt.assert_equal(walks, [[1, 6, 11, 5], [7, 17, 6, 1], [15, 33, 31, 9]])


def test_karate_club_random_walk_missing_connections(binary_karate_club_data):
    cl = client.MemoryGraph(binary_karate_club_data, [0, 1])
    walks = cl.random_walk(
        node_ids=np.array([2, 9, 13], dtype=np.int64),
        edge_types=1,
        walk_len=2,
        p=2,
        q=0.5,
        seed=1,
        default_node=-1,
    )

    npt.assert_equal(walks, [[2, -1, -1], [9, -1, -1], [13, -1, -1]])


def test_karate_club_random_walk_no_repetition_in_default_values(
    binary_karate_club_data,
):
    cl = client.MemoryGraph(binary_karate_club_data, [0, 1])
    walks = []
    for _ in range(2):
        walks.append(
            cl.random_walk(
                node_ids=np.array([1, 7, 15], dtype=np.int64),
                edge_types=0,
                walk_len=3,
                p=2,
                q=0.5,
            )
        )

    assert not np.array_equal(walks[0], walks[1])


def test_karate_club_random_walk_statistical(binary_karate_club_data):
    cl = client.MemoryGraph(binary_karate_club_data, [0, 1])
    walk_len = 3
    minibatch_size = 10
    actual_counts = [{} for _ in range(walk_len)]
    random.seed(27)
    for _ in range(10000):
        walks = cl.random_walk(
            node_ids=np.ones(minibatch_size, dtype=np.int64),
            edge_types=0,
            walk_len=walk_len,
            p=2,
            q=0.5,
            seed=random.getrandbits(64),
        )

        for input in range(minibatch_size):
            for step in range(walk_len):
                curr_node = walks[input][step + 1]
                if curr_node not in actual_counts[step]:
                    actual_counts[step][curr_node] = 0
                actual_counts[step][curr_node] += 1

    expected_counts = [
        {
            18: 6390,
            12: 6236,
            2: 6201,
            14: 6319,
            13: 6263,
            9: 6227,
            5: 6302,
            3: 6148,
            6: 6286,
            11: 6300,
            7: 6318,
            8: 6330,
            32: 6267,
            4: 6165,
            20: 6217,
            22: 6031,
        },
        {
            2: 14624,
            1: 20658,
            22: 663,
            3: 5541,
            4: 8217,
            31: 2929,
            34: 8680,
            7: 3995,
            11: 3858,
            17: 5557,
            5: 3945,
            8: 2179,
            25: 1130,
            33: 3832,
            14: 2237,
            6: 3946,
            28: 915,
            18: 654,
            29: 2092,
            10: 920,
            26: 1176,
            13: 1122,
            9: 469,
            20: 661,
        },
        {
            22: 3002,
            5: 3288,
            2: 5170,
            33: 1930,
            1: 8975,
            9: 3245,
            27: 553,
            7: 6885,
            20: 3541,
            34: 4111,
            3: 6619,
            6: 6895,
            17: 2458,
            14: 5671,
            28: 1920,
            4: 4323,
            16: 963,
            13: 2596,
            23: 870,
            12: 1102,
            8: 4821,
            19: 958,
            18: 2886,
            24: 1879,
            21: 909,
            10: 1181,
            32: 3047,
            11: 3322,
            29: 1198,
            31: 2937,
            15: 898,
            25: 611,
            26: 326,
            30: 910,
        },
    ]
    for step in range(len(expected_counts)):
        for node, count in expected_counts[step].items():
            assert actual_counts[step][node] == count


def test_karate_club_random_walk_single_server(binary_karate_club_data):
    s = server.Server(binary_karate_club_data, [0, 1], "localhost:9997")
    cl = client.DistributedGraph(["localhost:9997"])
    walks = cl.random_walk(
        node_ids=np.array([1, 7, 15], dtype=np.int64),
        edge_types=0,
        walk_len=3,
        p=2,
        q=0.5,
        seed=2,
    )

    npt.assert_equal(walks, [[1, 7, 1, 4], [7, 5, 11, 6], [15, 34, 27, 34]])
    s.reset()


def test_karate_club_random_walk_multiple_servers(binary_karate_club_data):
    s1 = server.Server(binary_karate_club_data, [0], "localhost:9996")
    s2 = server.Server(binary_karate_club_data, [1], "localhost:9995")
    cl = client.DistributedGraph(["localhost:9996", "localhost:9995"])
    walks = cl.random_walk(
        node_ids=np.array([1, 7, 15], dtype=np.int64),
        edge_types=0,
        walk_len=3,
        p=2,
        q=0.5,
        seed=3,
    )

    npt.assert_equal(walks, [[1, 20, 34, 32], [7, 5, 11, 1], [15, 34, 32, 1]])
    s1.reset()
    s2.reset()


if __name__ == "__main__":
    sys.exit(
        pytest.main(
            [__file__, "--junitxml", os.environ["XML_OUTPUT_FILE"], *sys.argv[1:]]
        )
    )
