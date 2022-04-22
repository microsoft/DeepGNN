# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
from deepgnn.graph_engine import DeepGNNDataset, GraphEngineBackend
from torch.utils.data import IterableDataset
from typing import Callable


class TorchDeepGNNDataset(IterableDataset, DeepGNNDataset):
    """TorchDeepGNNDataset is used by pytorch models,
    it derives from DeepGNNDataset and implements
    the IterableDataset class.
    """

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
        # parameters to initialize samplers
        **kwargs,
    ):
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
        # Note: when using multiple process to load the data in
        # parallel, each process should has its own copy of graph
        # client, otherwise there will be segmentfault error. Here
        # we return a None to postpone the initializeation of the
        # graph/sampler to __iter__.
        pass

    def init_sampler(self):
        # overide the base method to postpone the sampler
        # initialization to __iter__
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

    def __iter__(self):
        self._torch_init_sampler()
        return DeepGNNDataset.__iter__(self)
