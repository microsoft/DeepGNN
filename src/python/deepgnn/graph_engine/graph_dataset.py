# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from enum import Enum
from inspect import signature
from typing import Callable
from deepgnn.graph_engine._base import Graph
from deepgnn.graph_engine.backends.common import GraphEngineBackend
from deepgnn.graph_engine.prefetch import Generator
from deepgnn.graph_engine.samplers import BaseSampler
from deepgnn.graph_engine.backends.options import BackendOptions, GraphType


INVALID_NODE_ID = -1


class BackendType(Enum):
    """DeepGNN's graph engine backend type. Graph engine servers
    can be hosted on VM cluster or kubernetes.
    """

    SNARK = "snark"
    CUSTOM = "custom"

    def __str__(self):
        return self.value


class DeepGNNDataset:
    """Unified dataset shared by both TF and Torch.
    A typical dataset consists of:
        sampler which is used to sample seeds.
        query_fn which is a callback to generate batches.
    """

    class _DeepGNNDatasetIterator:
        def __init__(self, graph: Graph, sampler: BaseSampler, query_fn: Callable):
            self.graph = graph
            self.sampler = sampler
            self.query_fn = query_fn
            self.sampler_iter = iter(self.sampler)

        def __next__(self):
            inputs = next(self.sampler_iter)
            graph_tensor = self.query_fn(self.graph, inputs)
            return graph_tensor

        def __iter__(self):
            return self

    def __init__(
        self,
        sampler_class,
        query_fn: Callable = None,
        backend: GraphEngineBackend = None,
        num_workers: int = 1,
        worker_index: int = 0,
        batch_size: int = 1,
        epochs: int = 1,
        enable_prefetch: bool = False,
        collate_fn: Callable = None,
        # parameters to initialize samplers
        **kwargs,
    ):
        assert sampler_class is not None

        self.num_workers = num_workers
        self.sampler_class = sampler_class
        self.backend = backend
        self.worker_index = worker_index
        self.batch_size = batch_size
        self.epochs = epochs
        self.query_fn = query_fn
        self.enable_prefetch = enable_prefetch
        self.collate_fn = collate_fn
        self.kwargs = kwargs
        self.graph = None
        self.sampler = None

        self.init_graph_client()
        self.init_sampler()

    def init_graph_client(self):
        if self.backend is not None:
            self.graph = self.backend.graph

    def init_sampler(self):
        sig = signature(self.sampler_class.__init__)
        sampler_args = {}
        for key in sig.parameters:
            if key == "self":
                continue
            # Here we inject the graph dependency into sampler
            # if its __init__ needs a graph client,
            if sig.parameters[key].annotation == Graph:
                assert self.graph is not None
                sampler_args[key] = self.graph
            elif key in self.kwargs.keys():
                sampler_args[key] = self.kwargs[key]
            elif hasattr(self, key):
                sampler_args[key] = getattr(self, key)

        self.sampler = self.sampler_class(**sampler_args)

    def __iter__(self):
        if self.enable_prefetch:
            prefetch_size = (
                self.kwargs["prefetch_queue_size"]
                if "prefetch_queue_size" in self.kwargs
                else 10
            )
            max_parallel = (
                self.kwargs["prefetch_worker_size"]
                if "prefetch_worker_size" in self.kwargs
                else 2
            )

            return Generator(
                graph=self.graph,  # type: ignore
                sampler=self.sampler,
                model_query_fn=self.query_fn,  # type: ignore
                prefetch_size=prefetch_size,
                max_parallel=max_parallel,
                collate_fn=self.collate_fn,
            )
        else:
            return DeepGNNDataset._DeepGNNDatasetIterator(
                graph=self.graph, sampler=self.sampler, query_fn=self.query_fn
            )

    def __len__(self):
        return len(self.sampler)


def create_backend(backend_options: BackendOptions, is_leader: bool = False):
    backend_type = backend_options.backend
    if backend_type == BackendType.CUSTOM:
        backend_class = backend_options.custom_backendclass
    elif BackendType(backend_type) == BackendType.SNARK:
        if backend_options.graph_type == GraphType.LOCAL:
            assert backend_options.data_dir
            from deepgnn.graph_engine.backends.snark.client import (  # type: ignore
                SnarkLocalBackend as backend_class,
            )
        elif backend_options.graph_type == GraphType.REMOTE:
            from deepgnn.graph_engine.backends.snark.client import (  # type: ignore
                SnarkDistributedBackend as backend_class,
            )
    else:
        assert False, "Failed to determine a backend class"

    # worker with index 0 needs to check if the raw graph data is up
    # to date. If not, binary graph data need to be generated once again.
    return backend_class(options=backend_options, is_leader=is_leader)  # type: ignore
