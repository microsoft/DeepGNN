# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Synchronized server - client setup."""
from deepgnn.graph_engine.snark.distributed import Server, Client as DistributedClient
import ray


@ray.remote
class ServerWrapper(object):
    """Server wrapper."""

    def __init__(self):
        """Init server wrapper."""
        self.server = Server("localhost:9999", "/tmp/cora", 0, 1)

    def reset(self):
        """Reset server wrapper."""
        self.server.reset()

    def get_hostname(self):
        """Get hostname."""
        return self.server._hostname


@ray.remote
def start_servers(num: int):
    """Start N servers."""
    return [ServerWrapper.remote() for i in range(num)]  # type: ignore


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
