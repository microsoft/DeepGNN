# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Dataset implementation for torch models."""

import torch
from deepgnn.graph_engine import (
    DeepGNNDataset,
    GraphEngineBackend,
    Generator,
)
from torch.utils.data import IterableDataset
from typing import Callable, Union


class TorchDeepGNNDataset(IterableDataset, DeepGNNDataset):
    """Implementation of TorchDeepGNNDataset for use in a Torch Dataloader.

    TorchDeepGNNDataset initializes and executes a node or edge sampler given as
    sampler_class. For every batch of data requested, batch_size items are sampled
    from the sampler and passed to the given query_fn which pulls all necessaary
    information about the samples using the graph engine API. The output from
    the query function is passed to the trainer worker as the input to the
    model forward function.
    """

    def __init__(
        self,
        sampler_class,
        query_fn: Callable,
        backend: GraphEngineBackend = None,
        num_workers: int = 1,
        worker_index: int = 0,
        batch_size: int = 1,
        epochs: int = 1,
        enable_prefetch: bool = False,
        # parameters to initialize samplers
        **kwargs,
    ):
        """Initialize TorchDeepGNNDataset."""
        super(TorchDeepGNNDataset, self).__init__(
            sampler_class,
            query_fn,
            backend,
            num_workers,
            worker_index,
            batch_size,
            epochs,
            enable_prefetch,
            **kwargs,
        )

    def init_graph_client(self):
        """No-op function.

        When using multiple process to load the data in
        parallel, each process should has its own copy of graph
        client, otherwise there will be segmentfault error. Here
        we return a None to postpone the initializeation of the
        graph/sampler to __iter__.
        """
        pass

    def init_sampler(self):
        """No-op function.

        Overide the base method to postpone the sampler initialization to __iter__
        """
        return

    def _torch_init_sampler(self):
        # get the 'deep' copy of the graph.
        self.graph = self.backend.graph

        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            self.kwargs.update(
                {
                    "data_parallel_index": worker_info.id,
                    "data_parallel_num": worker_info.num_workers,
                }
            )
        super().init_sampler()

    def __iter__(self) -> Union[Generator, DeepGNNDataset._DeepGNNDatasetIterator]:
        """Create sampler and start iteration."""
        self._torch_init_sampler()
        return DeepGNNDataset.__iter__(self)
