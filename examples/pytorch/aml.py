# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
This example demonstrates how to use ray to submit a job to Azure ML using ray-on-aml.

Multi-node users need to follow these steps,
0. https://learn.microsoft.com/en-us/azure/machine-learning/how-to-secure-training-vnet?view=azureml-api-2&tabs=cli%2Crequired
1. Create a virtual network and add your compute to it.
2. Create a network security group.
3. Manually include pip packages shown in ray_on_aml + others required
"""
from typing import List, Any
from dataclasses import dataclass
import os
import numpy as np
import torch
import torch.nn.functional as F
from azureml.core import Dataset

import ray
import ray.train as train
from ray.train.torch import TorchTrainer
from ray.air import session
from ray.air.config import ScalingConfig, RunConfig

from deepgnn.graph_engine import Graph, graph_ops
from deepgnn.pytorch.modeling import BaseModel
from deepgnn.graph_engine.data.ppi import PPI
from deepgnn.graph_engine.snark.distributed import Server, Client as DistributedClient
from time import sleep
import socket


@ray.remote
class Cache:
    def __init__(self, size):
        self.cache = {}
        self.locks = [False for _ in range(size)]

    def set_lock(self, i):
        self.locks[i] = True
    
    def release_lock(self, i):
        self.locks[i] = False
    
    def wait_set(self):
        return all(self.locks)
    
    def wait_released(self):
        return all([not lock for lock in self.locks])

    def put(self, x, y):
        self.cache[x] = y

    def get(self, x):
        return self.cache.get(x)


def train_func(config: dict):
    """Training loop for ray trainer."""
    ppi = PPI(num_partitions=session.get_world_size())

    address = f"{ray._private.services.get_node_ip_address()}:999{session.get_world_rank()}"
    global_cache.put.remote(session.get_world_rank(), address)

    global_cache.set_lock.remote(session.get_world_rank())
    while not ray.get(global_cache.wait_set.remote()):
        sleep(1)
    address = ray.get(
        [global_cache.get.remote(i) for i in range(session.get_world_size())]
    )

    _ = Server(
        address[session.get_world_rank()], ppi.data_dir(), session.get_world_size(), 1
    )

    cl = DistributedClient(address)

    # TODO GAT
    sleep(10)

    global_cache.release_lock.remote(session.get_world_rank())
    while not ray.get(global_cache.wait_released.remote()):
        sleep(1)


if __name__ == "__main__":
    from azureml.core import Workspace
    from ray_on_aml.core import Ray_On_AML

    try:
        ws = Workspace.from_config()
        print(
            "Workspace name: " + ws.name,
            "Subscription id: " + ws.subscription_id,
            "Resource group: " + ws.resource_group,
            sep="\n",
        )
    except Exception:
        from azureml.core.authentication import ServicePrincipalAuthentication

        svc_pr = ServicePrincipalAuthentication(
            tenant_id=os.environ["TENANT_ID"],
            service_principal_id=os.environ["SERVICE_ID"],
            service_principal_password=os.environ["SERVICE_PASSWORD"],
        )
        ws = Workspace(
            subscription_id=os.environ["SUBSCRIPTION_ID"],
            resource_group=os.environ["RESOURCE_GROUP"],
            workspace_name=os.environ["WORKSPACE_NAME"],
            auth=svc_pr,
        )
    ray_on_aml = Ray_On_AML(ws=ws, compute_cluster="ray-cluster")

    try:
        ray = ray_on_aml.getRay(
            ci_is_head=True,
            num_node=3,
            pip_packages=[
                "ray[air]==2.4.0",
                "ray[data]==2.4.0",
                "azureml-mlflow==1.48.0",
                "torch==1.13.0",
                "deepgnn-torch",
            ],
        )
        num_workers = 2
        global_cache = Cache.remote(num_workers)

        trainer = TorchTrainer(
            train_func,
            train_loop_config={
                "ws_name": ws.name,
                "ws_subscription_id": ws.subscription_id,
                "ws_resource_group": ws.resource_group,
                "num_epochs": 10,
                "feature_idx": 0,
                "feature_dim": 1433,
                "label_idx": 1,
                "label_dim": 1,
                "num_classes": 7,
            },
            run_config=RunConfig(),
            scaling_config=ScalingConfig(
                num_workers=num_workers, placement_strategy="STRICT_SPREAD"
            ),
        )
        result = trainer.fit()
    except Exception as e:
        ray_on_aml.shutdown()
        raise e
    ray_on_aml.shutdown()
