# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Synchronized server - client setup."""
from deepgnn.graph_engine.snark.distributed import Server, Client as DistributedClient
import ray


@ray.remote
class ServerWrapper(object):
    def __init__(self):
        self.server = Server("localhost:9999", "/tmp/cora", 0, 1)

    def reset(self):
        self.server.reset()

    def get_hostname(self):
        return self.server._hostname


@ray.remote
def start_servers(num: int):
    return [ServerWrapper.remote() for i in range(num)]  # type: ignore


@ray.remote
def start_clients(servers, n_clients):
    return [
        DistributedClient([ray.get(server.get_hostname.remote()) for server in servers])
        for _ in range(n_clients)
    ]


@ray.remote
def train(servers, clients):

    for cl in clients:
        cl.reset()
    for s in servers:
        ray.get(s.reset.remote())
    return


servers = start_servers.bind(2)
clients = start_clients.bind(servers, 3)
output = train.bind(servers, clients)

from ray import workflow

print(workflow.run(output))
