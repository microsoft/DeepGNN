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
from deepgnn.graph_engine.data.citation import Cora
from deepgnn.graph_engine.snark.distributed import Server, Client as DistributedClient
from deepgnn.graph_engine.snark.synchronized import get_server_state


def test_simple_client_server_initialized_in_correct_order():
    working_dir = tempfile.TemporaryDirectory()
    Cora(working_dir.name)
    ray.shutdown()
    ray.init()
    server = Server("localhost:9999", working_dir.name, 0, 1)

    server_states = get_server_state()
    client = DistributedClient([state.get_hostname() for state in server_states])

    result = client.node_features(np.array([0, 1]), np.array([[1, 1]]), np.float32)
    npt.assert_almost_equal(result, np.array([[3.], [4.]], dtype=np.float32))

    client.reset()
    server.reset()
    ray.shutdown()


def test_client_initialization_timeout():
    # if a client is started without a server it should throw a timeout exception
    class MockClass:
        def reset(self):
            pass

    ray.shutdown()
    ray.init()
    working_dir = tempfile.TemporaryDirectory()
    with pytest.raises(TimeoutError):
        server_states = get_server_state()
    ray.shutdown()


def test_server_waits_for_client_to_stop():
    ray.shutdown()
    ray.init()
    working_dir = tempfile.TemporaryDirectory()
    Cora(working_dir.name)

    server = Server("localhost:9998", working_dir.name, 0, 1)
    server_states = get_server_state()
    client = DistributedClient([state.get_hostname() for state in server_states])

    with ThreadPoolExecutor() as executor:
        def server_done():
            server.reset()
        executor.submit(server_done)

    result = client.node_features(np.array([0, 1]), np.array([[1, 1]]), np.float32)
    npt.assert_almost_equal(result, np.array([[3.], [4.]], dtype=np.float32))

    for state in server_states:
        state.reset()
    client.reset()
    
    with pytest.raises(Exception):
        result = client.node_features(np.array([0, 1]), np.array([[1, 1]]), np.float32)
    
    ray.shutdown()


if __name__ == "__main__":
    sys.exit(
        pytest.main(
            [__file__, "--junitxml", os.environ["XML_OUTPUT_FILE"], *sys.argv[1:]]
        )
    )
