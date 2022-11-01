# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import pytest
import tempfile
import sys

import numpy as np
import numpy.testing as npt

from deepgnn.graph_engine.data.citation import Cora
from deepgnn.graph_engine.snark.distributed import start_distributed_backend


@pytest.fixture()
def cora_graph():
    output_dir = tempfile.TemporaryDirectory()
    dataset = Cora(output_dir.name)
    yield output_dir.name
    output_dir.cleanup()


def test_start_distributed_backend(cora_graph):
    start_distributed_backend(["127.0.0.1:9000"], cora_graph, 0, 0, 1)
    from time import sleep

    sleep(1)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, *sys.argv[1:]]))
