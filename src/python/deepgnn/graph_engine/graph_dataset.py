# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Common classes shared between TF and torch."""

from enum import Enum
from inspect import signature
from typing import Callable, Union
from deepgnn.graph_engine._base import Graph
from deepgnn.graph_engine.backends.common import GraphEngineBackend
from deepgnn.graph_engine.prefetch import Generator
from deepgnn.graph_engine.samplers import BaseSampler
from deepgnn.graph_engine.backends.options import BackendOptions, GraphType


INVALID_NODE_ID = -1


class BackendType(Enum):
    """Backend types for DeepGNN's graph engine.

    Graph engine servers can be hosted on VM cluster or kubernetes.
    """

    SNARK = "snark"
    CUSTOM = "custom"

    def __str__(self):
        """Convert enum to string."""
        return self.value


class DeepGNNDataset:
    """Unified dataset shared by both TF and Torch.

    DeepGNNDataset initializes and executes a node or edge sampler given as
    sampler_class. For every batch of data requested, batch_size items are sampled
    from the sampler and passed to the given query_fn which pulls all necessaary
    information about the samples using the graph engine API. The output from
    the query function is passed to the trainer worker as the input to the
    model forward function.
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
        sampler_class: BaseSampler,
        query_fn: Callable,
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
        """Initialize DeepGNN dataset."""
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
        self.graph: Graph
        self.sampler: BaseSampler

        self.init_graph_client()
        self.init_sampler()

    def init_graph_client(self):
        """Create graph client."""
        if self.backend is not None:
            self.graph = self.backend.graph

    def init_sampler(self):
        """Create sampler."""
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

    def __iter__(self) -> Union[Generator, _DeepGNNDatasetIterator]:
        """Create an iterator for graph."""
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
                graph=self.graph,
                sampler=self.sampler,
                model_query_fn=self.query_fn,
                prefetch_size=prefetch_size,
                max_parallel=max_parallel,
                collate_fn=self.collate_fn,
            )
        else:
            return DeepGNNDataset._DeepGNNDatasetIterator(
                graph=self.graph, sampler=self.sampler, query_fn=self.query_fn
            )

    def __len__(self) -> int:
        """Return number of elements in the sampler."""
        return len(self.sampler)


def create_backend(
    backend_options: BackendOptions, is_leader: bool = False
) -> GraphEngineBackend:
    """Entry function to initialize backends."""
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
