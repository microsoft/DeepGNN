# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from time import sleep
import ray


@ray.remote
class ServerState(object):
    def __init__(self, hostname):
        self.hostname = hostname
        self._clients = 0

    def get_hostname(self):
        return self.hostname

    def add_client(self):
        self._clients += 1

    def reset(self):
        self._clients -= 1

    def safe_to_terminate(self):
        return self._clients == 0


class ServerStateWrapped:
    def __init__(self, server_state):
        self.server_state = server_state

    def get_hostname(self):
        return ray.get(self.server_state.get_hostname.remote())

    def add_client(self):
        return ray.get(self.server_state.add_client.remote())

    def reset(self):
        return ray.get(self.server_state.reset.remote())

    def safe_to_terminate(self):
        return ray.get(self.server_state.safe_to_terminate.remote())


def set_server_state(hostname, index, namespace):
    try:
        ray.init(address="auto", ignore_reinit_error=True)
        return ServerState.options(name=f"server_{index}", namespace=namespace).remote(hostname)
    except ConnectionError:
        return None


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
    for state in server_states:
        state.add_client()
    return server_states
