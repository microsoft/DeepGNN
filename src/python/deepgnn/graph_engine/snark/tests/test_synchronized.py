# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import sys
from concurrent.futures.thread import ThreadPoolExecutor
import multiprocessing as mp
import tempfile

import pytest
import ray

from deepgnn.graph_engine.snark.distributed import Server, Client as DistributedClient, get_server_state


def test_simple_client_server_initialized_in_correct_order():
    server_event = mp.Event()

    class MockServer:
        def __init__(self):
            server_event.set()

        def reset(self):
            assert server_event.is_set()

    client_event = mp.Event()

    class MockClient:
        def __init__(self):
            client_event.set()

        def reset(self):
            assert client_event.is_set()

    working_dir = tempfile.TemporaryDirectory()
    ray.init()
    server = Server(working_dir.name, 0, None, MockServer)
    server_states = get_server_state()
    client = DistributedClient([state.get_hostname() for state in server_states])
    server_event.wait(1)
    client_event.wait(1)
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
    server_event = mp.Event()

    class MockServer:
        def __init__(self):
            server_event.set()

        def reset(self):
            assert server_event.is_set()

    client_event = mp.Event()

    class MockClient:
        def __init__(self):
            client_event.set()

        def reset(self):
            assert client_event.is_set()

    ray.init()
    working_dir = tempfile.TemporaryDirectory()
    server = Server(working_dir.name, 0, None, MockServer)
    server_states = get_server_state()
    client = DistributedClient([state.get_hostname() for state in server_states])
    server_event.wait()
    server_finished_event = mp.Event()
    client.client
    client_event.wait()
    with ThreadPoolExecutor() as executor:

        def server_done():
            server.reset()
            server_finished_event.set()

        executor.submit(server_done)
        client.reset()
    server_finished_event.wait()
    ray.shutdown()


if __name__ == "__main__":
    sys.exit(
        pytest.main(
            [__file__, "--junitxml", os.environ["XML_OUTPUT_FILE"], *sys.argv[1:]]
        )
    )
