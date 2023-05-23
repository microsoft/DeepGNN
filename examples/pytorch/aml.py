# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
This example demonstrates how to use ray to submit a job to Azure ML using ray-on-aml.

Multi-node users need to follow these steps,
0. https://learn.microsoft.com/en-us/azure/machine-learning/how-to-secure-training-vnet?view=azureml-api-2&tabs=cli%2Crequired
1. Create a virtual network and add your compute to it.
2. Create a network security group.
3. Manually include pip packages shown in ray_on_aml + others required
4. For multi-node, this script needs to be run in azure ml terminal not local.
"""
from time import sleep

import ray
from ray.train.torch import TorchTrainer
from ray.air import session
from ray.air.config import ScalingConfig, RunConfig

from deepgnn.graph_engine.data.ppi import PPI
from deepgnn.graph_engine.snark.distributed import Server, Client as DistributedClient


@ray.remote
class ServerLock:
    """Lock for server."""

    def __init__(self, size):
        """Initialize server lock."""
        self.cache = {}
        self.locks = [False for _ in range(size)]

    def set_lock(self, i):
        """Set lock i."""
        self.locks[i] = True

    def release_lock(self, i):
        """Release lock i."""
        self.locks[i] = False

    def wait_set(self):
        """See if all locks set."""
        return all(self.locks)

    def wait_released(self):
        """See if all locks released."""
        return all([not lock for lock in self.locks])

    def put(self, x, y):
        """Put value y at x."""
        self.cache[x] = y

    def get(self, x):
        """Get value x."""
        return self.cache.get(x)


def set_lock(server_lock, rank, address):
    """Set lock at rank, then halt until all servers locked."""
    server_lock.put.remote(rank, address)
    server_lock.set_lock.remote(rank)
    while not ray.get(server_lock.wait_set.remote()):
        sleep(1)


def release_lock(server_lock, rank):
    """Release lock at rank, then halt until all servers released."""
    server_lock.release_lock.remote(session.get_world_rank())
    while not ray.get(server_lock.wait_released.remote()):
        sleep(1)


def train_func(config: dict):
    """Training loop for ray trainer."""
    ppi = PPI(num_partitions=session.get_world_size())

    address = (
        f"{ray._private.services.get_node_ip_address()}:999{session.get_world_rank()}"
    )

    set_lock(server_lock, session.get_world_rank(), address)

    address = ray.get(
        [server_lock.get.remote(i) for i in range(session.get_world_size())]
    )

    Server(
        address[session.get_world_rank()], ppi.data_dir(), session.get_world_size(), 1
    )

    cl = DistributedClient(address)

    # TODO Replace these 2 lines with a model
    print(cl)
    sleep(10)

    release_lock(server_lock, session.get_world_rank())


if __name__ == "__main__":
    import azureml
    from azureml.core import Workspace
    from ray_on_aml.core import Ray_On_AML

    aml = True
    try:
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
    except azureml.exceptions._azureml_exception.UserErrorException:
        aml = False
        ray.init()
        ray_on_aml = type("Ray_On_AML", (object,), {"shutdown": (lambda: None)})

    try:
        num_workers = 2
        server_lock = ServerLock.remote(num_workers)  # type: ignore

        trainer = TorchTrainer(
            train_func,
            train_loop_config={
                "num_epochs": 10,
                "feature_idx": 0,
                "feature_dim": 1433,
                "label_idx": 1,
                "label_dim": 1,
                "num_classes": 7,
            },
            run_config=RunConfig(),
            scaling_config=ScalingConfig(
                num_workers=num_workers,
                placement_strategy="STRICT_SPREAD" if aml else "PACK",
            ),
        )
        result = trainer.fit()
    except Exception as e:
        ray_on_aml.shutdown()
        raise e
    ray_on_aml.shutdown()
