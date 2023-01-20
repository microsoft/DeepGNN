# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Snark districuted client implementation."""
from itertools import repeat
from typing import List, Dict, Union, Tuple
import logging
import tempfile
from time import sleep
import ray
import deepgnn.graph_engine.snark.client as client
import deepgnn.graph_engine.snark.server as server
import deepgnn.graph_engine.snark.local as ge_snark
from deepgnn import get_logger
from deepgnn.graph_engine.snark.meta import download_meta


class Client(ge_snark.Client):
    """Distributed client."""

    def __init__(
        self,
        servers: Union[str, List[str]],
        ssl_cert: str = None,
        grpc_options: List[Tuple[str, str]] = None,
    ):
        """Init snark client to wrapper around ctypes API of distributed graph."""
        self.logger = get_logger()
        self.logger.info(f"servers: {servers}. SSL: {ssl_cert}")
        if isinstance(servers, str):
            servers = [servers]
        self._servers = servers
        self._ssl_cert = ssl_cert
        self.graph = client.DistributedGraph(
            servers, ssl_cert, grpc_options=grpc_options
        )
        self.node_samplers: Dict[str, client.NodeSampler] = {}
        self.edge_samplers: Dict[str, client.EdgeSampler] = {}
        self.logger.info(
            f"Loaded distributed snark client. Node counts: {self.graph.meta.node_count_per_type}. Edge counts: {self.graph.meta.edge_count_per_type}"
        )

    def __reduce__(self):
        """On serialize reload as new client."""

        def deserialize(*args):
            get_logger().setLevel(logging.ERROR)
            return Client(*args)

        return deserialize, (self._servers, self._ssl_cert)


@ray.remote
class ServerState(object):
    def __init__(self, hostname):
        self.hostname = hostname

    def get_hostname(self):
        return self.hostname

class ServerStateWrapped:
    def __init__(self, server_state):
        self.server_state = server_state

    def get_hostname(self):
        return ray.get(self.server_state.get_hostname.remote())


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
        namespace: str = "deepgnn",
    ):
        """Init snark server."""
        self._hostname = hostname
        self._ssl_cert = ssl_cert

        self._init_args = (hostname, data_path, index, total_shards)

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
        partitions = list(
            zip(
                repeat(data_path, partition_count),
                range(index, partition_count, total_shards),
            )
        )
        self.server = server.Server(
            data_path,
            partitions,
            hostname,
            ssl_config,  # type: ignore
            storage_type,
            config_path,
            stream,  # type: ignore
        )

        try:
            ray.init(address="auto", ignore_reinit_error=True)
            self.actor = ServerState.options(name=f"server_{index}", namespace=namespace).remote(hostname)
        except ConnectionError:
            pass

    def reset(self):
        """Reset server."""
        if self.server is not None:
            self.server.reset()

    def __reduce__(self):
        """On serialize reload as new client."""

        def deserialize(*args):
            get_logger().setLevel(logging.ERROR)
            return Server(*args)

        return deserialize, self._init_args


def get_server_state(num_servers: int = 1, timeout: int = 30, connect_delay: int = 5, namespace: str = "deepgnn"):
    ray.init(address="auto", ignore_reinit_error=True)
    server_states = []
    for i in range(num_servers):
        print(f"Connecting to Server {i}...")
        for _ in range(timeout // connect_delay):
            try:
                server_state = ray.get_actor(f"server_{i}", namespace=namespace)
                break
            except ValueError:
                sleep(connect_delay)
        else:
            raise TimeoutError(f"Failed to connect to server {i}!")
        print(f"Connected to Server {i}.")
        server_states.append(ServerStateWrapped(server_state))
    return server_states
