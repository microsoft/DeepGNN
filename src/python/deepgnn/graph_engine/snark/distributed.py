# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Snark districuted client implementation.

Synchronized wrappers for client and server to use as a backend for GE.
The idea is to use sync files to wait until all servers are created or every client is closed.
Every backend knows how many servers are going to be started through command line arguments,
so clients wait until that number of server sync files appear in a sync folder(usually a model path).
To synchronize shutdown we track every client created via a sync file with snark_{client_id}.client pattern.
Servers will stop only after all these files are deleted by clients.
"""
from typing import Optional, List, Dict, Any, Callable
import tempfile
import concurrent
from concurrent.futures import ThreadPoolExecutor
import os
import pathlib
import time
import multiprocessing as mp
import threading
import glob
import socket

from deepgnn.graph_engine.snark.client import PartitionStorageType
import deepgnn.graph_engine.snark.client as client
import deepgnn.graph_engine.snark.server as server
import deepgnn.graph_engine.snark.local as ge_snark
from deepgnn import get_logger
from deepgnn.graph_engine.snark.meta import download_meta


class DistributedClient(ge_snark.Client):
    """Distributed client."""

    def __init__(self, servers: List[str], ssl_cert: str = None):
        """Init snark client to wrapper around ctypes API of distributed graph."""
        self.logger = get_logger()
        self.logger.info(f"servers: {servers}. SSL: {ssl_cert}")
        self.graph = client.DistributedGraph(servers, ssl_cert)
        self.node_samplers: Dict[str, client.NodeSampler] = {}
        self.edge_samplers: Dict[str, client.EdgeSampler] = {}
        self.logger.info(
            f"Loaded distributed snark client. Node counts: {self.graph.meta.node_count_per_type}. Edge counts: {self.graph.meta.edge_count_per_type}"
        )


class Server:
    """Distributed server."""

    def __init__(
        self,
        hostname: str,
        data_path: str,
        index: int,
        total_shards: int,
        ssl_key: str = None,
        ssl_root: str = None,
        ssl_cert: str = None,
        storage_type: client.PartitionStorageType = client.PartitionStorageType.memory,
        config_path: str = "",
        stream: bool = False,
    ):
        """Init snark server."""
        temp_dir = tempfile.TemporaryDirectory()
        temp_path = temp_dir.name
        meta_path = download_meta(data_path, temp_path, config_path)

        with open(meta_path, "r") as meta:
            # TODO(alsamylk): expose graph metadata reader in snark.
            # Based on snark.client._read_meta() method
            skip_lines = 7
            for _ in range(skip_lines):
                meta.readline()
            partition_count = int(meta.readline())

        ssl_config = None
        if ssl_key is not None:
            ssl_config = {
                "ssl_key": ssl_key,
                "ssl_root": ssl_root,
                "ssl_cert": ssl_cert,
            }
        partitions = list(range(index, partition_count, total_shards))
        self.server = server.Server(
            data_path,
            partitions,
            hostname,
            ssl_config,  # type: ignore
            storage_type,
            config_path,
            stream,  # type: ignore
        )

    def reset(self):
        """Reset server."""
        if self.server is not None:
            self.server.reset()


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
        self._client = None
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
        self._server = None
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

    def join(self, timeout: float = None):
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
        """Initialize server.

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


# Helper functions and classes to automatically determine servers in a cluster:
# Every worker will write out it's primary address with a port via _AddressFinder,
# and a server index obtained from environment variables/passed through command line.
# Classes will act essentially as file servers. After all these files are created
# the _AddressReader class will read all addresses and feed to the backend so snark client
# can connect to every GE instance.
def _get_host() -> str:
    # Find a primary address of a compute instance
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        try:
            s.connect(("10.255.255.255", 1))
            return s.getsockname()[0]
        except Exception:
            return "localhost"


# A class to write out worker's primary address to the path.
# The class must be wrapped with SynchronizedServer.
class _AddressFinder:
    def __init__(self, path: str, id: int, server_idx: int):
        get_logger().debug("Starting address finder")
        self.file_name = os.path.join(path, str(id) + ".servername")
        with open(self.file_name, "w+", newline=os.linesep) as file:
            host = _get_host()
            port = self._get_free_port(host)
            file.write(host + ":" + str(port))
            file.write(os.linesep)
            file.write(str(server_idx))

    def _get_free_port(self, host: str) -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((host, 0))
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            return s.getsockname()[1]

    def reset(self):
        os.remove(self.file_name)


# Read all addresses from path.
# _AddressReader must be wrapped by SynchronizedClient to wait all servers.
class _AddressReader:
    def __init__(self, path: str):
        self._servers: List[str] = []
        for file_name in glob.glob(os.path.join(path, "*.servername")):
            with open(file_name, "r", newline=os.linesep) as f:
                # First line is the address and the second - server index.
                raw = f.readlines()
                assert len(raw) == 2
                server_idx = int(raw[1])
                if server_idx >= 0:
                    self._servers.append(raw[0].rstrip(os.linesep))

    def servers(self) -> List[str]:
        return self._servers

    def reset(self):
        return


def start_distributed_backend(
    server_hostnames: List[str],
    data_dir: str,
    server_index: int,
    client_rank: int,
    world_size: int,
    sync_dir: Optional[str] = None,
    ge_start_timeout: int = 30,
    ssl_cert=None,
    storage_type: PartitionStorageType = PartitionStorageType.memory,
    config_path: str = "",
    stream: bool = True,
):
    """Initialize a snark client connected to to a GE server.

    Parameters
    ----------
    server_index: int Index of server, use -1 to skip starting GE in worker.
    client_rank: int Index of client.
    world_size: int Number of clients to initialize.
    sync_dir: str, default="." Dir to store lock and address files.
    """
    if sync_dir is None or len(sync_dir) == 0:
        sync_dir = os.path.join(".", "sync")
        try:
            os.mkdir(sync_dir)
        except FileExistsError:
            pass
        get_logger().debug(f"Defaulting to {sync_dir} to synchronize GE and workers.")
    lock_dir = os.path.join(sync_dir, "workers")
    addr_dir = os.path.join(sync_dir, "addresses")
    try:
        os.mkdir(sync_dir)
    except FileExistsError:
        pass
    try:
        os.mkdir(addr_dir)
    except FileExistsError:
        pass

    for i, hostname in enumerate(server_hostnames):
        server = SynchronizedServer(
            lock_dir,
            server_index,
            ge_start_timeout,
            Server,  # _AddressFinder
            hostname,  # addr_dir
            data_dir,  # client_rank
            server_index,
            len(server_hostnames),
            ssl_cert=ssl_cert,
            storage_type=storage_type,
            config_path=config_path,
            stream=stream,
        )

    client = SynchronizedClient(  # type: ignore
        sync_dir,
        client_rank,
        world_size,
        ge_start_timeout,
        # Below 3 or _AddressReader, address_folder,
        DistributedClient,
        server_hostnames,
        ssl_cert,
    )

    client.reset()
    server.reset()
    return client, server
