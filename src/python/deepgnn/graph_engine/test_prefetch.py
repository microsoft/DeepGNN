# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import unittest.mock as mock
import math
import pytest
import numpy as np

from deepgnn.graph_engine._base import Graph
from deepgnn.graph_engine.samplers import RangeNodeSampler, BaseSampler
from deepgnn.graph_engine import prefetch


@pytest.fixture(scope="module")
def create_graph():
    g = Graph()
    yield g


class ExceptionSampler(BaseSampler):
    def __init__(self, exception_msg):
        self.msg = exception_msg

    def __iter__(self):
        return self

    def __next__(self):
        raise Exception(self.msg)


def test_handle_sampler_exception(create_graph):
    s = ExceptionSampler("test sampler exception")
    model = mock.MagicMock()
    model.sampler = s

    gen = prefetch.Generator(create_graph, model.sampler, model.query)
    num_iterations = 0

    with pytest.raises(Exception):
        for _ in gen():
            num_iterations += 1

    assert num_iterations == 0
    gen.join()


def test_handle_query_exception(create_graph):
    model = mock.MagicMock()
    attrs = {"query.side_effect": Exception("test query exception")}
    model = mock.MagicMock(**attrs)
    model.sampler = RangeNodeSampler(0, 10, 6, 0, 1, backfill_id=-1)
    gen = prefetch.Generator(create_graph, model.sampler, model.query)

    with pytest.raises(Exception):
        for _ in gen():
            pass

    gen.join()


def test_generator_join_with_break(create_graph):
    s = RangeNodeSampler(0, 1024, 32, 0, 1, backfill_id=-1)
    model = mock.MagicMock()
    model.sampler = s

    gen = prefetch.Generator(create_graph, model.sampler, model.query)
    num_iterations = 0

    for i, _ in enumerate(gen):
        if i >= 20:
            break
        num_iterations += 1

    gen.join()
    # make sure the join will not block the process.
    assert True


def test_generator_join(create_graph):
    for total_size in range(1, 50):
        for batch_size in range(1, total_size):
            s = RangeNodeSampler(0, total_size, batch_size, 0, 1, backfill_id=-1)
            attrs = {"query.return_value": {"inputs": np.ndarray([1], np.int64)}}
            model = mock.MagicMock(**attrs)
            model.sampler = s

            gen = prefetch.Generator(create_graph, model.sampler, model.query)
            num_iterations = 0
            for _, minibatch in enumerate(gen):
                num_iterations += 1

            assert num_iterations == math.ceil(total_size / batch_size)
            gen.join()
