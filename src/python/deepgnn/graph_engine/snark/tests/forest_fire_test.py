# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import sys
import pytest
import os
import random
import networkx as nx
import numpy as np
import deepgnn.graph_engine.snark.preprocess.sampler.forest_fire as ff
import deepgnn.graph_engine.snark.preprocess.sampler.metric as metr


def setup_function():
    random.seed(5)
    np.random.seed(5)


@pytest.fixture
def graph_choice_one():
    g = nx.scale_free_graph(1000, seed=100)
    return g


# Ensure that forestfire produces a correctly sized sample graph
def test_node_count(graph_choice_one):
    sample = ff.forest_fire(graph_choice_one, 300)
    assert len(list(sample.nodes)) == 300


# Densification power law
def test_T1_1(graph_choice_one):
    sample = ff.forest_fire(graph_choice_one, 100)
    densification_input = metr.densification(graph_choice_one)
    densification_sample = metr.densification(sample)
    assert densification_input == pytest.approx(
        1.1118193002297947
    ) and densification_sample == pytest.approx(1.061925820483543)


# Densification power law
def test_T1_2(graph_choice_one):
    sample = ff.forest_fire(graph_choice_one, 400)
    densification_input = metr.densification(graph_choice_one)
    densification_sample = metr.densification(sample)
    assert densification_input == pytest.approx(
        1.1118193002297947
    ) and densification_sample == pytest.approx(1.0934021696257126)


# Densification power law
def test_T1_3(graph_choice_one):
    sample = ff.forest_fire(graph_choice_one, 800)
    densification_input = metr.densification(graph_choice_one)
    densification_sample = metr.densification(sample)
    assert densification_input == pytest.approx(
        1.1118193002297947
    ) and densification_sample == pytest.approx(1.0878273867110992)


# Densification power law
def test_T1_4(graph_choice_one):
    sample = ff.forest_fire(graph_choice_one, 250)
    densification_input = metr.densification(graph_choice_one)
    densification_sample = metr.densification(sample)
    assert densification_input == pytest.approx(
        1.1118193002297947
    ) and densification_sample == pytest.approx(1.0917904898554778)


# Densification power law
def test_T1_5(graph_choice_one):
    sample = ff.forest_fire(graph_choice_one, 800)
    densification_input = metr.densification(graph_choice_one)
    densification_sample = metr.densification(sample)
    assert densification_input == pytest.approx(
        1.1118193002297947
    ) and densification_sample == pytest.approx(1.0878273867110992)


# Shrinking diameter
def test_T2_1(graph_choice_one):
    sample = ff.forest_fire(graph_choice_one, 200)
    diameter_input = metr.diameter(graph_choice_one)
    diameter_sample = metr.diameter(sample)
    assert diameter_input == 7 and diameter_sample == 9


# Shrinking diameter
def test_T2_2(graph_choice_one):
    sample = ff.forest_fire(graph_choice_one, 300)
    diameter_input = metr.diameter(graph_choice_one)
    diameter_sample = metr.diameter(sample)
    assert diameter_input == 7 and diameter_sample == 7


# Shrinking diameter
def test_T2_3(graph_choice_one):
    sample = ff.forest_fire(graph_choice_one, 500)
    diameter_input = metr.diameter(graph_choice_one)
    diameter_sample = metr.diameter(sample)
    assert diameter_input == 7 and diameter_sample == 7


# Normalized largest connected component
def test_T3(graph_choice_one):
    sample = ff.forest_fire(graph_choice_one, 300)
    input_largest_connected_component = metr.largest_connected_component(
        graph_choice_one
    )
    sample_largest_connected_component = metr.largest_connected_component(sample)
    assert input_largest_connected_component == pytest.approx(
        0.046
    ) and sample_largest_connected_component == pytest.approx(0.0866666)


if __name__ == "__main__":
    sys.exit(
        pytest.main(
            [__file__, "--junitxml", os.environ["XML_OUTPUT_FILE"], *sys.argv[1:]]
        )
    )
