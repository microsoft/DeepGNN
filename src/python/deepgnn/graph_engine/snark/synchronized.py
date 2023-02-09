# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Synchronized server - client setup."""
from deepgnn.graph_engine.snark.distributed import Server, Client as DistributedClient
import ray


@ray.remote
class ServerWrapper(object):
    """Server wrapper."""

    def __init__(self, hostname, *server_args, **server_kwargs):
        """Init server wrapper."""
        self._hostname = hostname
        self.server = Server(hostname, *server_args, **server_kwargs)

    def reset(self):
        """Reset server wrapper."""
        self.server.reset()

    def get_hostname(self):
        """Get hostname."""
        return self._hostname


@ray.remote
def start_servers(
    hostname: str, data_dir: str, n_servers: int, n_partitions: int = 1, **server_kwargs
):
    """Start N servers."""
    return [
        ServerWrapper.remote(  # type: ignore
            hostname, data_dir, i, n_partitions // n_servers, **server_kwargs
        )
        for i in range(n_servers)
    ]


@ray.remote
def start_clients(servers, n_clients):
    """Start N clients."""
    return [
        DistributedClient([ray.get(server.get_hostname.remote()) for server in servers])
        for _ in range(n_clients)
    ]


@ray.remote
def train(servers, clients, fn):
    """Execute train fn."""
    result = fn(clients)

    for cl in clients:
        cl.reset()
    for s in servers:
        ray.get(s.reset.remote())
    return result
