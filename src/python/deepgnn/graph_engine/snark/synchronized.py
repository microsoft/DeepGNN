# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Any, List
import ray
from deepgnn.graph_engine.snark.distributed import Server, Client as DistributedClient


class SynchronizedServer(Server):
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


class SynchronizedClient(DistributedClient):
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

    def reset(self):
        """Disconnect client from servers and unload it from memory."""
        with self._lock:
            if self._client is not None:
                self._client.reset()



if __name__ == "__main__":
    from ray import workflow

    #@ray.remote
    def start_server(sync_path: str, path: str, partitions: List[int], hostname: str):
        # TODO create sync file
        return Server(hostname, path, 0, len(partitions))#path, partitions, hostname, delayed_start=True)

    def start_client(sync_path, hostname) -> List[float]:
        # TODO wait until sync files created in path
        return DistributedClient(hostname)

    sync_path = "/tmp/sync"
    path = "/tmp/cora"
    partitions = [0]
    hostname = "localhost:9999"
    server = start_server(sync_path, path, partitions, hostname)
    client_obj = start_client(sync_path, hostname)

    # TODO locks to actually start client
    
    client = workflow.run(client_obj)
    print(server)
    print(client)


    # TODO synchronize to wait for servers to be allowed to shutdown
