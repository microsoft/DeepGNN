# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from dataclasses import dataclass
import os
import sys

import numpy as np
import pytest

from deepgnn.graph_engine.data.cora import CoraFull
import deepgnn.graph_engine.snark.server as server
import deepgnn.graph_engine.snark.distributed as distributed


def weighted_sample(graph, batches):
    total = 0
    for batch in batches:
        nodes = graph.sample_neighbors(
            strategy="byweight",
            nodes=batch,
            edge_types=np.array([0], dtype=np.int32),
            count=50,
        )[0]
        total += len(nodes)
    return total


def sample(graph, batches, alpha, epsilon):
    total = 0
    for batch in batches:
        nodes = graph.sample_neighbors(
            strategy="ppr-go",
            nodes=batch,
            edge_types=np.array([0], dtype=np.int32),
            count=50,
            alpha=alpha,
            eps=epsilon,
        )[0]
        total += len(nodes)
    return total


@dataclass
class BenchmarkData:
    graph: str
    inputs: np.array
    returned_nodes_count: int = 0


@pytest.fixture(scope="session")
def dataset():
    graph = CoraFull()
    batch_size = 256
    nodes = np.arange(
        graph.NUM_NODES + (batch_size - graph.NUM_NODES % batch_size),
        dtype=np.int64,
    )
    batches = nodes.reshape(-1, batch_size)
    return BenchmarkData(graph, batches, len(nodes))


def test_ppr_on_cora_distributed(benchmark, dataset):
    s = server.Server(
        dataset.graph.data_dir(), partitions=[0], hostname="localhost:50051"
    )
    c = distributed.Client(["localhost:50051"])
    result = benchmark(
        sample,
        c,
        dataset.inputs,
        0.85,
        0.0001,
    )
    c.reset()
    s.reset()
    assert result == dataset.returned_nodes_count


def test_ppr_on_cora_in_memory(benchmark, dataset):
    result = benchmark(sample, dataset.graph, dataset.inputs, 0.85, 0.0001)
    assert result == dataset.returned_nodes_count


def test_weighted_on_cora_in_memory(benchmark, dataset):
    result = benchmark(weighted_sample, dataset.graph, dataset.inputs)
    assert result == dataset.returned_nodes_count


if __name__ == "__main__":
    sys.exit(
        pytest.main(
            [__file__, "--junitxml", os.environ["XML_OUTPUT_FILE"], *sys.argv[1:]]
        )
    )
