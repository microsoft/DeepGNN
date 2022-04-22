# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import sys
import pytest
import os
import networkx as nx
import random
import deepgnn.graph_engine.snark.preprocess.sampler.metric as metr


@pytest.fixture
def graph_choice_one():
    random.seed(5)
    g = nx.scale_free_graph(1000, seed=100)
    return g


@pytest.fixture
def graph_choice_two():
    g = nx.star_graph(100).to_directed()
    return g


def test_densification_one(graph_choice_one):
    d = metr.densification(graph_choice_one)
    assert d == pytest.approx(1.11181930002297)


def test_densification_two(graph_choice_two):
    d = metr.densification(graph_choice_two)
    assert d == pytest.approx(1.1480344548346)


def test_effective_diameter_one(graph_choice_one):
    d = metr.diameter(graph_choice_one)
    assert d == 7


def test_effective_diameter_two(graph_choice_two):
    d = metr.diameter(graph_choice_two)
    assert d == 2


def test_largest_eigenvalue_one(graph_choice_one):
    d = metr.max_adjacency(graph_choice_one)
    assert d == pytest.approx(52.7454614388038)


def test_largest_eigenvalue_two(graph_choice_two):
    d = metr.max_adjacency(graph_choice_two)
    assert d == pytest.approx(10)


if __name__ == "__main__":
    sys.exit(
        pytest.main(
            [__file__, "--junitxml", os.environ["XML_OUTPUT_FILE"], *sys.argv[1:]]
        )
    )
