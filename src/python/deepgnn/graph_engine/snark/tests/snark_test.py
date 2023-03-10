# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import random
import json
import tempfile
import sys
import logging
import io

import networkx as nx
import numpy as np
import numpy.testing as npt
import deepgnn.graph_engine.snark.convert as convert
import deepgnn.graph_engine.snark.decoders as decoders
import deepgnn.graph_engine.snark.server as server
import pytest

from deepgnn.graph_engine.snark.local import Client as LocalClient
import deepgnn.graph_engine.snark.distributed as distributed
from deepgnn.graph_engine._base import SamplingStrategy
from deepgnn.graph_engine.snark.meta_merger import merge_metadata_files
from deepgnn.graph_engine.snark.meta import Meta


def caveman_data(partitions: int = 1, worker_count: int = 1, output_dir: str = ""):
    random.seed(246)
    g = nx.connected_caveman_graph(30, 12)
    nodes = []
    data = [""] * worker_count  # every worker process one sub file
    for node_id in g:
        # Set weights for neighbors
        nbs = {}
        for nb in nx.neighbors(g, node_id):
            nbs[nb] = 1.0

        node = {
            "node_weight": 1,
            "node_id": node_id,
            "node_type": 0,
            "uint64_feature": {},
            "float_feature": {"0": [node_id, random.random()]},
            "binary_feature": {},
            "edge": [
                {
                    "src_id": node_id,
                    "dst_id": nb,
                    "edge_type": 0,
                    "weight": 1.0,
                    "uint64_feature": {},
                    "float_feature": {},
                    "binary_feature": {},
                }
                for nb in nx.neighbors(g, node_id)
            ],
        }
        data[node_id % worker_count] += json.dumps(node) + "\n"
        nodes.append(node)

    working_dir = tempfile.TemporaryDirectory()
    for i in range(worker_count):
        raw_file = working_dir.name + f"/data{i}.json"
        with open(raw_file, "w+") as f:
            f.write(data[i])
    [
        convert.MultiWorkersConverter(
            graph_path=working_dir.name,
            partition_count=partitions,
            output_dir=output_dir,
            worker_index=n,
            worker_count=worker_count,
            decoder=decoders.JsonDecoder(),
        ).convert()
        for n in range(worker_count)
    ]

    merge_metadata_files(output_dir)


@pytest.fixture(scope="module")
def memory_graph(request):
    random.seed(23)
    output_dir = tempfile.TemporaryDirectory()

    partitions = 2 if not hasattr(request, "param") else request.param[0]
    worker_count = 1 if not hasattr(request, "param") else request.param[1]
    caveman_data(partitions, worker_count, output_dir.name)

    yield LocalClient(output_dir.name, partitions=[0, 1])
    output_dir.cleanup()


# parametize format: (partion_count, worker_count)
@pytest.mark.parametrize("memory_graph", [(2, 1), (2, 2)], indirect=True)
def test_snark_backend_local_graph_node_features(memory_graph):
    values = memory_graph.node_features(
        np.array([1], dtype=np.int64),
        np.array([[0, 2]], dtype=np.int32),
        np.float32,
    )

    assert values.shape == (1, 2)
    npt.assert_almost_equal([[1, 0.516677]], values, decimal=4)


def test_snark_backend_local_graph_sample_nodes(memory_graph):
    random.seed(23)
    result = memory_graph.sample_nodes(5, 0, SamplingStrategy.Weighted)

    nodes, types = memory_graph.sample_nodes(
        5, np.array([0], dtype=np.int32), SamplingStrategy.Weighted
    )

    npt.assert_equal([122, 82, 210, 317, 219], result)
    npt.assert_equal([36, 276, 83, 295, 259], nodes)
    npt.assert_equal([0] * 5, types)


def test_snark_backend_local_graph_sample_edges(memory_graph):
    random.seed(42)
    values = memory_graph.sample_edges(2, 0, SamplingStrategy.Weighted)
    npt.assert_equal([[174, 168, 0], [129, 128, 0]], values)


@pytest.mark.xfail
@pytest.mark.parametrize("memory_graph", [(2, 1), (2, 2)], indirect=True)
def test_snark_backend_local_graph_sample_neighbors(memory_graph):
    random.seed(42)
    nbs, wt, tp, _ = memory_graph.sample_neighbors(
        np.array([1, 3], dtype=np.int64), 0, 3
    )
    npt.assert_equal([[11, 6, 9], [5, 2, 11]], nbs)
    npt.assert_equal([[1, 1, 1], [1, 1, 1]], wt)
    npt.assert_equal([[0, 0, 0], [0, 0, 0]], tp)


@pytest.fixture(scope="module")
def distributed_graph(request):
    output_dir = tempfile.TemporaryDirectory()
    partitions = 2 if not hasattr(request, "param") else request.param[0]
    worker_count = 1 if not hasattr(request, "param") else request.param[1]
    caveman_data(partitions, worker_count, output_dir.name)

    s1 = server.Server(output_dir.name, [(output_dir.name, 0)], "localhost:11234")
    s2 = server.Server(output_dir.name, [(output_dir.name, 1)], "localhost:11235")
    cl = distributed.Client(["localhost:11234", "localhost:11235"])

    yield cl

    cl.reset()
    s2.reset()
    s1.reset()


@pytest.mark.parametrize("distributed_graph", [(2, 1), (2, 2)], indirect=True)
def test_snark_backend_distributed_graph_node_features(distributed_graph):
    values = distributed_graph.node_features(
        np.array([1], dtype=np.int64),
        np.array([[0, 2]], dtype=np.int32),
        np.float32,
    )
    assert values.shape == (1, 2)
    npt.assert_almost_equal([[1, 0.516677]], values, decimal=4)


def test_snark_backend_distributed_graph_sample_nodes(distributed_graph):
    random.seed(23)
    result = distributed_graph.sample_nodes(5, 0, SamplingStrategy.Weighted)

    nodes, types = distributed_graph.sample_nodes(
        5, np.array([0], dtype=np.int32), SamplingStrategy.Weighted
    )

    npt.assert_equal([52, 146, 30, 238, 125], result)
    npt.assert_equal([110, 150, 36, 241, 21], nodes)
    npt.assert_equal([0] * 5, types)


def test_snark_backend_distributed_graph_sample_edges(distributed_graph):
    random.seed(42)
    values = distributed_graph.sample_edges(2, 0, SamplingStrategy.Weighted)
    npt.assert_equal([[237, 238, 0], [21, 13, 0]], values)


@pytest.mark.parametrize("distributed_graph", [(2, 1), (2, 2)], indirect=True)
def test_snark_backend_distributed_graph_sample_neighbors(distributed_graph):
    random.seed(42)
    nbs, wt, tp, _ = distributed_graph.sample_neighbors(
        np.array([1, 3], dtype=np.int64), 0, 3
    )

    npt.assert_equal([[4, 9, 4], [1, 8, 11]], nbs)
    npt.assert_equal([[1, 1, 1], [1, 1, 1]], wt)
    npt.assert_equal([[0, 0, 0], [0, 0, 0]], tp)


@pytest.mark.parametrize("distributed_graph", [(2, 1), (2, 2)], indirect=True)
def test_snark_backend_distributed_graph_random_walk(distributed_graph):
    random.seed(42)
    nodes = distributed_graph.random_walk(
        np.array([1, 3], dtype=np.int64), 0, 3, p=0.1, q=2
    )
    npt.assert_equal([[1, 3, 1, 3], [3, 5, 7, 5]], nodes)


@pytest.mark.parametrize("distributed_graph", [(2, 1), (2, 2)], indirect=True)
def test_snark_backend_distributed_graph_features_missing_from_graph(distributed_graph):
    string_stream = io.StringIO()
    stream_handler = logging.StreamHandler(string_stream)
    distributed_graph.logger.disabled = False
    distributed_graph.logger.addHandler(stream_handler)
    values = distributed_graph.node_features(
        np.array([1], dtype=np.int64),
        np.array([[2, 3]], dtype=np.int32),
        np.float32,
    )
    log_record = string_stream.getvalue()
    assert (
        log_record.find(
            "Requesting feature with id #2 that is larger than number of the node features 1 in the graph"
        )
        >= 0
    )

    assert values.shape == (1, 3)
    npt.assert_equal(values, [[0, 0, 0]])

    values = distributed_graph.edge_features(
        np.array([[1, 2, 0]], dtype=np.int64),
        np.array([[1, 2]], dtype=np.int32),
        np.float32,
    )
    log_record = string_stream.getvalue()
    assert (
        log_record.find(
            "Requesting feature with id #1 that is larger than number of the edge features 0 in the graph"
        )
        > 0
    )

    assert values.shape == (1, 2)
    npt.assert_equal(values, [[0, 0]])


def test_meta_version_message():
    working_dir = tempfile.TemporaryDirectory()
    meta_file = working_dir.name + f"/meta.json"
    with open(meta_file, "w+") as f:
        f.writelines(['{"node_count": 10, "edge_count": 10}'])
    with pytest.raises(KeyError) as excinfo:
        Meta(working_dir.name)
        assert "KeyError: 'binary_data_version'" in str(excinfo.value)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, *sys.argv[1:]]))
