# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import sys
from concurrent.futures.thread import ThreadPoolExecutor
import multiprocessing as mp
import tempfile

import pytest
import numpy as np
import numpy.testing as npt
import ray
from ray import workflow
from deepgnn.graph_engine.data.citation import Cora
from deepgnn.graph_engine.snark.distributed import Server, Client as DistributedClient
from deepgnn.graph_engine.snark.synchronized import start_servers, start_clients, train


def test_simple_client_server_initialized_in_correct_order():
    working_dir = tempfile.TemporaryDirectory()
    Cora(working_dir.name)

    def fn(clients):
        cl = clients[0]
        result = cl.node_features(np.array([0, 1]), np.array([[1, 1]]), np.float32)
        return result

    servers = start_servers.bind(1)
    clients = start_clients.bind(servers, 1)
    output = train.bind(servers, clients, fn)
    result = workflow.run(output)

    npt.assert_almost_equal(result, np.array([[3.0], [4.0]], dtype=np.float32))


if __name__ == "__main__":
    sys.exit(
        pytest.main(
            [__file__, "--junitxml", os.environ["XML_OUTPUT_FILE"], *sys.argv[1:]]
        )
    )
