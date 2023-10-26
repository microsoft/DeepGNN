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
                "0": [123, 64, 12],
            },
            "edge": [{"src_id": 9, "dst_id": 0, "edge_type": 0, "weight": 1}],
        },
        {
            "node_id": 0,
            "node_type": 1,
            "node_weight": 1,
            "uint64_feature": {},
            "float_feature": {"0": [1], "1": [-0.03, -0.04]},
            "binary_feature": {"3": "abcd"},
            "edge": [{"src_id": 0, "dst_id": 5, "edge_type": 0, "weight": 1}],
        },
        {
            "node_id": 5,
            "node_type": 2,
            "node_weight": 1,
            "binary_feature": {"3": "abcd", "5": "something long"},
            "edge": [{"src_id": 5, "dst_id": 9, "edge_type": 0, "weight": 1}],
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
        Path(output.name), 2, Counter(), JsonDecoder(), watermark=-1
    )
    convert.MultiWorkersConverter(
        graph_path=data_name,
        partition_count=1,
        output_dir=output.name,
        decoder=JsonDecoder(),
        dispatcher=d,
        watermark=-1,
    ).convert()

    yield output.name


@pytest.fixture(params=["inmemory", "distributed"])
def graph_dataset(multi_partition_graph_data, request):
    import shutil

    print(f"log: {multi_partition_graph_data}")
    shutil.copytree(multi_partition_graph_data, "/tmp/update_test", dirs_exist_ok=True)
    if request.param == "inmemory":
        yield client.MemoryGraph(multi_partition_graph_data, [0, 1])
    else:
        address = [f"localhost:{find_free_port()}", f"localhost:{find_free_port()}"]
        s1 = server.Server(multi_partition_graph_data, [0], address[0])
        s2 = server.Server(multi_partition_graph_data, [1], address[1])
        yield client.DistributedGraph(address)
        s1.reset()
        s2.reset()


def test_simple_feature_update(graph_dataset):
    f = graph_dataset.node_features([9], [[0, 2]], dtype=np.float32)
    npt.assert_array_equal([[123, 64]], f)

    # new sizes contains lengths of values written to GE.
    new_sizes = graph_dataset.update_node_features(
        [9],
        [[0, 2]],
        values=np.array([[3, 4]], dtype=np.float32),
    )
    npt.assert_array_equal([[2]], new_sizes)
    f = graph_dataset.node_features([9], [[0, 2]], dtype=np.float32)
    npt.assert_array_equal([[3, 4]], f)


def test_feature_update_graph_nonexistent_node(graph_dataset):
    # Fetch features for a node that doesn't exist
    f = graph_dataset.node_features([13], [[0, 2]], dtype=np.float32)
    npt.assert_array_equal([[0, 0]], f)

    # Assign some values to that node's features
    sizes = graph_dataset.update_node_features(
        [13],
        [[0, 2]],
        values=np.array([[3, 4]], dtype=np.float32),
    )
    # Nothing should be written to the graph
    npt.assert_array_equal([[0]], sizes)

    # Verify that the node and it's features still don't exist
    f = graph_dataset.node_features([13], [[0, 2]], dtype=np.float32)
    npt.assert_array_equal([[0, 0]], f)


def test_feature_update_for_different_dimensions(graph_dataset):
    # Node 9 has only 1 feature of dimension 3.
    # Node 0 has 2 features of dimension 1 and 2
    # Node 5 doesn't have any features
    f = graph_dataset.node_features([9, 0, 5], [[0, 3], [1, 1]], dtype=np.float32)
    npt.assert_array_almost_equal([[123, 64, 12, 0], [1, 0, 0, -0.03], [0, 0, 0, 0]], f)

    # New sizes contains lengths of values written to GE.
    # We will write 2 features, both with dimension 2.
    # If nodes(with id 9) have longer features already in memory, then values after dimension 2 will remain intact.
    # For nodes(with id 0) with shorter features, we will write only up to their length.
    # If nodes(with id 5) don't have any features, we will not do anything.
    new_sizes = graph_dataset.update_node_features(
        [9, 0, 5],
        [[0, 2], [1, 2]],
        values=np.array(
            [[21, 22, 23, 24], [31, 32, 33, 34], [40, 41, 42, 43]], dtype=np.float32
        ),
    )
    npt.assert_array_equal([[2, 0], [1, 2], [0, 0]], new_sizes)
    f = graph_dataset.node_features([9, 0, 5], [[0, 3], [1, 2]], dtype=np.float32)
    npt.assert_array_almost_equal(
        [[21, 22, 12, 0, 0], [31, 0, 0, 33, 34], [0, 0, 0, 0, 0]], f
    )


if __name__ == "__main__":
    sys.exit(
        pytest.main(
            [__file__, "--junitxml", os.environ["XML_OUTPUT_FILE"], *sys.argv[1:]]
        )
    )
