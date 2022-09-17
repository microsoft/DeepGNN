# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Snark districuted client implementation."""
from typing import List, Dict
import tempfile

import deepgnn.graph_engine.snark.client as client
import deepgnn.graph_engine.snark.server as server
import deepgnn.graph_engine.snark.local as ge_snark
from deepgnn import get_logger
from deepgnn.graph_engine.snark.meta import download_meta
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
import json
import os
import glob
import threading
from typing import Optional, Tuple, List
import socket

from deepgnn.logging_utils import get_logger
from deepgnn.graph_engine.backends.options import BackendOptions
from deepgnn.graph_engine.backends.common import GraphEngineBackend
from deepgnn.graph_engine.snark.local import Client as LocalClient
from deepgnn.graph_engine.snark.distributed import (
    Client as DistributedClient,
    Server as Server,
)
from deepgnn.graph_engine.backends.snark.synchronized import (
    SynchronizedClient,
    SynchronizedServer,
)


class Client(ge_snark.Client):
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


class SnarkDistributedBackend(GraphEngineBackend):
    """SnarkDistributedBackend is used to initialize a snark client connected to to a GE server."""

    def __init__(self, options: BackendOptions, is_leader: bool = False):
        """Initialize a distributed backend."""
        self._server: Optional[SynchronizedServer] = None
        # Short circuit the simplest case: connect to existing backends.
        if options.skip_ge_start:
            class PassThrough:
                def __init__(self, options: BackendOptions) -> None:
                    self.lock = threading.Lock()
                    self.servers: List[str] = options.servers
                    self.ssl_cert: Optional[str] = (
                        options.ssl_cert if options.enable_ssl else None
                    )
                    self._client: Optional[DistributedClient] = None

                @property
                def client(self):
                    with self.lock:
                        if self._client is None:
                            self._client = DistributedClient(
                                self.servers, self.ssl_cert
                            )
                    return self._client

                def reset(self):
                    if self._client is not None:
                        self._client.reset()

            self._client = PassThrough(options)
            return

        # First start all servers and create snark_(server_id).server file
        # model dir to flag readines.

        server_index, client_rank, world_size = self._get_server_index_client_rank(
            options
        )
        assert (
            server_index is not None
            and client_rank is not None
            and world_size is not None
        ), "Server index should be always initialized. Use -1 to skip starting GE in a worker"

        # Try our best to automatically detect GE servers if they were not passed as options.
        self._check_servers_option(options, world_size, client_rank, server_index)

        sync_dir = options.sync_dir
        if sync_dir is None or len(sync_dir) == 0:
            sync_dir = os.path.join(options.model_dir, "sync")
            get_logger().debug(
                f"Defaulting to {sync_dir} to synchronize GE and workers."
            )

        os.makedirs(sync_dir, exist_ok=True)
        if server_index >= 0:
            self._server = SynchronizedServer(
                sync_dir,
                server_index,
                options.ge_start_timeout,
                Server,
                options.servers[server_index],
                options.data_dir,
                server_index,
                len(options.servers),
                ssl_cert=options.ssl_cert,
                storage_type=options.storage_type,
                config_path=options.config_path,
                stream=options.stream,
            )
        assert client_rank is not None, "Client rank should be always initialized"
        self._client = SynchronizedClient(  # type: ignore
            sync_dir,
            client_rank,
            len(options.servers),
            options.ge_start_timeout,
            DistributedClient,
            options.servers,
            options.ssl_cert,
        )

    def _check_servers_option(
        self,
        options: BackendOptions,
        world_size: int,
        client_rank: int,
        server_idx: int,
    ):
        if options.servers is not None and len(options.servers) > 0:
            return
        sync_folder = os.path.join(options.model_dir, "sync", "workers")
        address_folder = os.path.join(options.model_dir, "sync", "addresses")
        os.makedirs(sync_folder, exist_ok=True)
        os.makedirs(address_folder, exist_ok=True)
        server = SynchronizedServer(
            sync_folder,
            client_rank,
            options.ge_start_timeout,
            _AddressFinder,
            address_folder,
            client_rank,
            server_idx,
        )
        client = SynchronizedClient(
            sync_folder,
            client_rank,
            world_size,
            options.ge_start_timeout,
            _AddressReader,
            address_folder,
        )
        options.servers = client.client.servers()
        client.reset()
        server.reset()
        get_logger().warning(
            f"GE servers were not specified. Starting them on {options.servers}"
        )

    # Return a server index to use in the host list. Negative value means don't start server.
    def _get_server_index_client_rank(
        self, options: BackendOptions
    ) -> Tuple[Optional[int], Optional[int], Optional[int]]:
        # Calculate number of GE instances
        if options.num_ge == 0 and (
            options.servers is None or len(options.servers) == 0
        ):
            # TODO(alsamylk): provide a better metadata reader in snark
            with open(os.path.join(options.data_dir, "meta.txt"), "r") as meta:
                # Use a simple heuristic: number of original partitions, but cap it at 4
                # to avoid extreme cases. Partition count is written on 7th line.
                lines = meta.readlines()
                options.num_ge = min(int(lines[7]), 4)

        # Use explicitly set index if it was provided
        if options.server_idx is not None:
            world_size = (
                len(options.servers)
                if options.servers is not None and len(options.servers) > 0
                else options.num_ge
            )
            get_logger().info(
                f"Using command line arguments to synchronize GE, server:{options.server_idx}, client: {options.client_rank}, world_size: {world_size}"
            )
            return options.server_idx, options.client_rank, world_size

        # Iterate through ddp and mpi
        server_index, client_rank, world_size = self._try_ddp(options)
        if server_index is not None and client_rank is not None:
            get_logger().info(
                f"Synchronizing GE via DDP, server: {server_index}, client: {client_rank}, world_size: {world_size}"
            )
            return server_index, client_rank, world_size

        server_index, client_rank, world_size = self._try_mpi_or_horovod()
        if server_index is not None and client_rank is not None:
            get_logger().info(
                f"Synchronizing GE via MPI, server: {server_index}, client: {client_rank}, world_size: {world_size}"
            )
            return server_index, client_rank, world_size

        server_index, client_rank, world_size = self._try_tf_cmd_args(options)
        if server_index is not None and client_rank is not None:
            get_logger().info(
                f"Synchronizing GE via TF PS, server: {server_index}, client: {client_rank}, world_size: {world_size}"
            )
            return server_index, client_rank, world_size

        server_index, client_rank, world_size = self._try_tf_config()
        if server_index is not None and client_rank is not None:
            get_logger().info(
                f"Synchronizing GE via TF PS, server: {server_index}, client: {client_rank}, world_size: {world_size}"
            )
            return server_index, client_rank, world_size

        get_logger().error("Not synchronizing GE")
        return None, None, None

    def _try_ddp(
        self, options: BackendOptions
    ) -> Tuple[Optional[int], Optional[int], Optional[int]]:
        local_rank = os.getenv("LOCAL_RANK")
        if local_rank is None:
            return None, None, None

        world_size = os.getenv("WORLD_SIZE")
        assert world_size is not None

        local_size = os.getenv("LOCAL_WORLD_SIZE")
        assert local_size is not None
        world_rank = os.getenv("RANK")
        assert world_rank is not None

        server_index = (
            int(world_rank) // int(local_size) if int(local_rank) == 0 else -1
        )

        return server_index, int(world_rank), int(world_size)

    def _try_mpi_or_horovod(self) -> Tuple[Optional[int], Optional[int], Optional[int]]:
        local_rank_key = "OMPI_COMM_WORLD_LOCAL_RANK"
        local_size_key = "OMPI_COMM_WORLD_LOCAL_SIZE"
        world_rank_key = "OMPI_COMM_WORLD_RANK"
        world_size_key = "OMPI_COMM_WORLD_SIZE"

        if local_rank_key not in os.environ:
            # try horovod prefix first
            local_rank_key = "HOROVOD_LOCAL_RANK"
            local_size_key = "HOROVOD_LOCAL_SIZE"
            world_rank_key = "HOROVOD_RANK"
            world_size_key = "HOROVOD_SIZE"

        local_rank = os.getenv(local_rank_key)
        if local_rank is None:
            return None, None, None

        world_rank = os.getenv(world_rank_key)
        assert world_rank is not None

        local_size = os.getenv(local_size_key)
        assert local_size is not None

        # Keep a simple solution for uniform setup.
        # If we'll need a heterogeneous setup,
        # we can wait on servers to write their local sizes first
        # and then start them.
        server_index = (
            int(world_rank) // int(local_size) if int(local_rank) == 0 else -1
        )
        client_rank = int(world_rank)
        world_size = os.getenv(world_size_key)
        assert world_size is not None

        return server_index, client_rank, int(world_size)

    def _extract_server_index(self, workers: List[str], index: int) -> Optional[int]:
        curr_worker = workers[index]
        workers.sort()
        seen_hosts = set()
        for worker in workers:
            host = worker.rsplit(":", 1)[0]
            if curr_worker != worker:
                seen_hosts.add(host)
                continue
            server_index = -1 if host in seen_hosts else len(seen_hosts)
            return server_index
        return None

    def _try_tf_config(self) -> Tuple[Optional[int], Optional[int], Optional[int]]:
        tf_config_raw = os.getenv("TF_CONFIG")
        if tf_config_raw is None:
            return None, None, None
        tf_config = json.loads(tf_config_raw)
        if tf_config["task"]["type"].lower() != "worker":
            return -1, -1, 0
        workers: List[str] = tf_config["cluster"]["worker"]
        client_rank: int = tf_config["task"]["index"]
        return (
            self._extract_server_index(workers, client_rank),
            client_rank,
            len(workers),
        )

    def _try_tf_cmd_args(
        self, options: BackendOptions
    ) -> Tuple[Optional[int], Optional[int], Optional[int]]:
        if (
            not hasattr(options, "task_index")
            or options.task_index is None  # type: ignore
        ):  # type: ignore
            return None, None, None
        client_rank: int = options.task_index  # type: ignore
        if (
            not hasattr(options, "worker_hosts")
            or options.worker_hosts is None  # type: ignore
            or len(options.worker_hosts) == 0  # type: ignore
            or options.worker_hosts[0]  # type: ignore
            == ""  # Result of command line arg default value "".split(",")
        ):
            return None, None, None
        workers: List[str] = options.worker_hosts  # type: ignore

        return (
            self._extract_server_index(workers, client_rank),
            client_rank,
            len(workers),
        )

    @property
    def graph(self):
        """Return distributed graph client."""
        return self._client.client

    def close(self):
        """Stop clients and then servers."""
        if self._client is not None:
            self._client.reset()
        if self._server is not None:
            self._server.reset()
