# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Backend implementation of C++ graph engine."""
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


class SnarkLocalBackend(GraphEngineBackend):
    """SnarkLocalBackend is used to initialize an in-memory graph backend."""

    def __init__(self, options: BackendOptions, is_leader: bool = False):
        """Initialize backend with a local graph."""
        self._client = LocalClient(
            options.data_dir,
            options.partitions,
            options.storage_type,
            options.config_path,
            options.stream,
        )

    @property
    def graph(self):
        """Get graph."""
        return self._client

    def close(self):
        """Typically we don't want to unload graph in local mode, hence no-op."""
        pass


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
