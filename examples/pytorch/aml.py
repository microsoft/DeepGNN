# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
This example demonstrates how to use ray to submit a job to Azure ML using ray-on-aml.

Multi-node users need to follow these steps,
0., https://learn.microsoft.com/en-us/azure/machine-learning/how-to-secure-training-vnet?view=azureml-api-2&tabs=cli%2Crequired
1. Create a virtual network and add your compute to it.
2. Create a network security group.
3. Manually include pip packages shown in ray_on_aml + others required
4. For multi-node, this script needs to be run in azure ml terminal not local.
"""
import numpy as np
import numpy.testing as npt
import ray
from ray.train.torch import TorchTrainer
from ray.air import session
from ray.air.config import ScalingConfig, RunConfig

from deepgnn.graph_engine.data.cora import CoraFull
from deepgnn.graph_engine.snark.distributed import Server, Client as DistributedClient


@ray.remote
class ServerLock:
    """Lock for server."""

    def __init__(self, size):
        """Initialize server lock."""
        self.cache = {}
        self.locks = [0 for _ in range(size)]

    def set_lock(self, i):
        """Set lock i."""
        self.locks[i] = 1

    def release_lock(self, i):
        """Release lock i."""
        self.locks[i] = 2

    def wait_set(self):
        """See if all locks set."""
        return all([v == 1 for v in self.locks])

    def wait_released(self):
        """See if all locks released."""
        return all([v == 2 for v in self.locks])

    def put(self, x, y):
        """Put value y at x."""
        self.cache[x] = y

    def get(self, x):
        """Get value x."""
        return self.cache.get(x)


class ServerContext:
    """Server context."""

    def __init__(self, world_size: int, rank: int, address: str):
        """Initialize server context."""
        self.world_size = world_size
        self.rank = rank
        self.address = address
        self.server_lock = server_lock

    def __enter__(self):
        """Enter server context."""
        self.server_lock.put.remote(self.rank, self.address)
        self.server_lock.set_lock.remote(self.rank)
        while not ray.get(self.server_lock.wait_set.remote()):
            pass
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        """Exit server context."""
        self.server_lock.release_lock.remote(self.rank)
        while not ray.get(self.server_lock.wait_released.remote()):
            pass

    def get_address(self):
        """Return addresses of all servers connected."""
        return ray.get([self.server_lock.get.remote(i) for i in range(self.world_size)])


def train_func(config: dict):
    """Training loop for ray trainer."""
    cora = CoraFull(num_partitions=session.get_world_size())

    address = f"127.0.0.1:999{session.get_world_rank()}"

    Server(address, cora.data_dir(), session.get_world_rank(), 1)

    with ServerContext(
        session.get_world_size(), session.get_world_rank(), address
    ) as lock:
        cl = DistributedClient(lock.get_address())

        features = cl.node_features(np.array([2, 3]), np.array([[1, 1]]), np.float32)
        npt.assert_equal(
            features,
            np.array(
                [
                    [
                        1.0,
                    ],
                    [
                        2.0,
                    ],
                ]
            ),
        )


if __name__ == "__main__":
    """# Ray on AML setup
    import sys
    from azureml.core import Workspace
    from ray_on_aml.core import Ray_On_AML

    ws = Workspace.from_config()
    print(
        "Workspace name: " + ws.name,
        "Subscription id: " + ws.subscription_id,
        "Resource group: " + ws.resource_group,
        sep="\n",
    )

    ray_on_aml = Ray_On_AML(ws=ws, compute_cluster="ray-cluster")
    ray = ray_on_aml.getRay(
        ci_is_head=True,
        num_node=3,
        pip_packages=[
            "ray[air]",
            "ray[data]",
            "azureml-mlflow==1.48.0",
            "torch",
            "deepgnn-torch",
        ],
    )
    """

    ray.init(num_cpus=3)
    ray_on_aml = type("Ray_On_AML", (object,), {"shutdown": (lambda self: None)})()

    try:
        num_workers = 2
        server_lock = ServerLock.remote(num_workers)  # type: ignore

        trainer = TorchTrainer(
            train_func,
            run_config=RunConfig(),
            scaling_config=ScalingConfig(
                num_workers=num_workers,
            ),
        )
        result = trainer.fit()
    except Exception as e:
        ray_on_aml.shutdown()
        raise e
    ray_on_aml.shutdown()
