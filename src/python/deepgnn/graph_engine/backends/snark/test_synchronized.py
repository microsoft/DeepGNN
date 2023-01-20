# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from concurrent.futures.thread import ThreadPoolExecutor
import multiprocessing as mp
import tempfile

import pytest

import deepgnn.graph_engine.backends.snark.synchronized as synchronized


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
    server = synchronized.SynchronizedServer(working_dir.name, 0, None, MockServer)
    client = synchronized.SynchronizedClient(working_dir.name, 0, 1, None, MockClient)
    server_event.wait(1)
    client_event.wait(1)
    client.reset()
    server.reset()


def test_client_initialization_timeout():
    # if a client is started without a server it should throw a timeout exception
    class MockClass:
        def reset(self):
            pass

    working_dir = tempfile.TemporaryDirectory()
    with pytest.raises(TimeoutError):
        wrapper = synchronized.SynchronizedClient(
            working_dir.name, rank=0, num_servers=1, timeout=1, klass=MockClass
        )
        wrapper.client


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

    working_dir = tempfile.TemporaryDirectory()
    server = synchronized.SynchronizedServer(working_dir.name, 0, None, MockServer)
    client = synchronized.SynchronizedClient(working_dir.name, 0, 1, None, MockClient)
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
