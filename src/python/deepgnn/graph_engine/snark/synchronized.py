# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Synchronized server - client setup."""
from typing import Optional, List
from time import sleep
import ray


@ray.remote
class ServerState:
    """State of a single server."""

    def __init__(self, hostname: str):
        """Init server state."""
        self.hostname = hostname
        self._clients = 0

    def get_hostname(self) -> str:
        """Get hostname of server."""
        return self.hostname

    def add_client(self):
        """Add client to server."""
        self._clients += 1

    def reset(self):
        """Reset server state."""
        self._clients -= 1

    def safe_to_terminate(self) -> bool:
        """Is server safe to terminate?."""
        return self._clients == 0


class ServerStateWrapped:
    """Wrapped client facing state of a single server."""

    def __init__(self, server_state: ServerState):
        """Init wrapped server state."""
        self.server_state = server_state

    def get_hostname(self) -> str:
        """Get hostname of server."""
        return ray.get(self.server_state.get_hostname.remote())  # type: ignore

    def add_client(self):
        """Add client to server."""
        return ray.get(self.server_state.add_client.remote())  # type: ignore

    def reset(self):
        """Reset server state."""
        return ray.get(self.server_state.reset.remote())  # type: ignore

    def safe_to_terminate(self) -> bool:
        """Is server safe to terminate?."""
        return ray.get(self.server_state.safe_to_terminate.remote())  # type: ignore


def _set_server_state(
    hostname: str, index: int, namespace: str = "deepgnn"
) -> Optional[ServerState]:
    """Add server state to ray namespace."""
    try:
        ray.init(address="auto", ignore_reinit_error=True)
        return ServerState.options(name=f"server_{index}", namespace=namespace).remote(  # type: ignore
            hostname
        )
    except ConnectionError:
        return None


def get_server_state(
    num_servers: int = 1,
    timeout: int = 30,
    connect_delay: int = 5,
    namespace: str = "deepgnn",
) -> List[ServerStateWrapped]:
    """Pull server state from ray namespace."""
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
    for state in server_states:
        state.add_client()
    return server_states
