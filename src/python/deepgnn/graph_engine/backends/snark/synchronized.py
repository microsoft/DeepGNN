# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Synchronized wrappers for client and server to use as a backend for GE.

The idea is to use sync files to wait until all servers are created or every client is closed.
Every backend knows how many servers are going to be started through command line arguments,
so clients wait until that number of server sync files appear in a sync folder(usually a model path).
To synchronize shutdown we track every client created via a sync file with snark_{client_id}.client pattern.
Servers will stop only after all these files are deleted by clients.
"""
import concurrent
from concurrent.futures import ThreadPoolExecutor
import os
import pathlib
import time
import multiprocessing as mp
import threading
import glob

from typing import Optional, Any, Callable
from deepgnn.logging_utils import get_logger


def _create_lock_file(folder: str, index: int, extension: str):
    file_name = pathlib.Path(folder, f"snark_{index}.{extension}")

    # If the sync folder is not clean, we can't guarantee initialization
    # order of all servers/clients.
    assert not _check_lock_file(
        folder, index, extension
    ), f"Delete sync folder {folder} before starting training."

    with open(file_name, "w+") as f:
        f.write(str(os.getpid()))


def _check_lock_file(folder: str, index: int, extension: str):
    return os.path.exists(pathlib.Path(folder, f"snark_{index}.{extension}"))


def _delete_lock_file(folder: str, index: int, extension: str):
    path = pathlib.Path(folder, f"snark_{index}.{extension}")
    os.remove(path)


class SupportsReset:
    """Simple interface for client/server."""

    def reset(self) -> None:
        """Unload client/server from memory."""
        pass


class SynchronizedClient:
    """SynchronizedClient uses file system to synchronize create graph client only after every GE instance started.

    Servers appear in the `path` folder as files snark_#[0-n].server and client creation is delayed until these sync files appear.
    """

    def __init__(
        self,
        path: str,
        rank: int,
        num_servers: int,
        timeout: float,
        klass: Any,
        *args,
        **kwargs,
    ):
        """Initialize client."""
        self.rank = rank
        self.path = path
        self.original_pid = os.getpid()
        self.num_servers = num_servers
        self.timeout = timeout
        self._client: Optional[SupportsReset] = None
        self._ktr = lambda: klass(*args, **kwargs)
        self._lock = threading.Lock()
        _create_lock_file(self.path, self.rank, "client")

    @property
    def client(self):
        """Connect client to all servers and save it for future use."""
        # Use optimistic lock
        if self._client is not None:
            return self._client
        with self._lock:
            if self._client is not None:
                return self._client
            if self._wait_for_servers():
                self._client = self._ktr()
        if self._client is None:
            # Delete lock files to prevent servers to timeout.
            self.reset()
            raise TimeoutError(
                f"Failed to connect to all {self.num_servers} servers with client #{self.rank}"
            )
        return self._client

    def reset(self):
        """Disconnect client from servers and unload it from memory."""
        with self._lock:
            if self._client is not None:
                self._client.reset()
            if os.getpid() == self.original_pid:
                _delete_lock_file(self.path, self.rank, "client")

    def _wait_for_servers(self):
        class WaitForServer:
            def __init__(self, index: int, path: str):
                self.index = index
                self.path = path
                # A flag to drain the task queue.
                self.keep_going = True

            def __call__(self):
                while self.keep_going and not _check_lock_file(
                    self.path, self.index, "server"
                ):
                    get_logger().info(f"Waiting for server #{self.index} {self.path}")
                    time.sleep(1)

        result = True
        with ThreadPoolExecutor() as executor:
            tasks = [
                WaitForServer(index, self.path) for index in range(self.num_servers)
            ]
            tasks_with_futures = [(task, executor.submit(task)) for task in tasks]
            for task, future in tasks_with_futures:
                try:
                    future.result(timeout=self.timeout)
                    get_logger().info(
                        f"Client #{os.getpid()} found server #{task.index}"
                    )
                except concurrent.futures.TimeoutError:
                    get_logger().info(f"Client #{task.index} failed to wait on server")
                    task.keep_going = False
                    result = False
        return result


class _ServerProcess(mp.Process):
    def __init__(self, sync_path: str, index: int, klass: Callable, *args, **kwargs):
        super(_ServerProcess, self).__init__()
        self.id = index
        self.sync_path = sync_path
        self._ktr = lambda: klass(*args, **kwargs)
        self._server: Optional[SupportsReset] = None
        self._stop_event = mp.Event()

    def _wait_for_clients(self, timeout: Optional[float]):
        class WaitForClients:
            def __init__(self, path: str):
                self.path = path
                # A flag to drain the task queue.
                self.keep_going = True

            def __call__(self):
                while (
                    self.keep_going
                    and len(glob.glob(os.path.join(self.path, "*.client"))) > 0
                ):
                    active_clients = glob.glob(os.path.join(self.path, "*.client"))
                    get_logger().info(
                        f"Servers are waiting for {active_clients} to close connection."
                    )
                    time.sleep(1)

        with ThreadPoolExecutor() as executor:
            task = WaitForClients(self.sync_path)
            try:
                future = executor.submit(task)
                future.result(timeout=timeout)
                get_logger().info(
                    f"Server #{self.id} done waiting for clients to disconnect"
                )
            except concurrent.futures.TimeoutError:
                get_logger().info("Server timed out waiting for clients to disconnect")
                task.keep_going = False

    def run(self):
        """Start a GE server."""
        self._server = self._ktr()
        _create_lock_file(self.sync_path, self.id, "server")
        self._stop_event.wait()
        get_logger().info(f"Shutting down server #{self.id}")

    def join(self, timeout: Optional[float] = None):
        """Stop the GE server."""
        self._wait_for_clients(timeout)
        self._stop_event.set()
        if self._server is not None:
            self._server.reset()


class SynchronizedServer:
    """SynchronizedServer uses file system to delay server deletion on shutdown.

    Until all client sync files are deleted from the `sync_path` folder, the servers will keep running.
    """

    def __init__(
        self, sync_path: str, index: int, timeout: float, klass: Any, *args, **kwargs
    ):
        """
        Initialize server.

        A backend might be forked(e.g. by pytorch DDP), so we need to start a separate process to protect mutexes.
        """
        self.sync_path = sync_path
        self.id = index
        self.timeout = timeout
        self.original_pid = os.getpid()
        self._server_process = _ServerProcess(sync_path, index, klass, *args, **kwargs)
        self._server_process.start()

    def reset(self):
        """Unload server from memory."""
        self._server_process.join(timeout=self.timeout)
        if os.getpid() == self.original_pid:
            _delete_lock_file(self.sync_path, self.id, "server")
