****************************
Torch Dataset Usage
****************************

In this guide we create a few Torch Datasets with various sampling strategies then we go through ways of setting a train test split.

Generate Dataset
================

.. code-block:: python

    >>> from typing import List, Tuple, Any, Iterator
    >>> import random
    >>> import numpy as np
    >>> import torch
    >>> from torch.utils.data import Dataset, DataLoader, Sampler
    >>> from deepgnn.graph_engine import SamplingStrategy
    >>> from deepgnn.graph_engine.snark.local import Client
    >>> random.seed(0)
    >>> torch.manual_seed(0)
    <torch._C.Generator object at 0x...>

    >>> from deepgnn.graph_engine.data.citation import Cora
    >>> Cora("/tmp/cora")  # Generate Cora dataset (Train: 140, Valid: 500, Test: 1000)
    <deepgnn.graph_engine.data.citation.Cora object at 0x...>

Simple Cora Dataset
================

.. code-block:: python

    >>> class DeepGNNDataset(Dataset):
    ...     """Cora dataset with base torch sampler."""
    ...     def __init__(self, data_dir: str, node_types: List[int], feature_meta: List[int]):
    ...         self.g = Client(data_dir, [0, 1])
    ...         self.node_types = np.array(node_types)
    ...         self.feature_meta = feature_meta
    ...         self.count = self.g.node_count(self.node_types)
    ...
    ...     def __len__(self):
    ...         return self.count
    ... 
    ...     def __getitem__(self, idx: int) -> Tuple[Any, Any]:
    ...         return self.g.node_features([idx], np.array([self.feature_meta]), feature_type=np.float32), torch.Tensor([0])

    >>> dataset = DeepGNNDataset("/tmp/cora", [0, 1, 2], [1, 50])
    >>> train_dataset, test_dataset = torch.utils.data.random_split(dataset, [int(.8 * len(dataset)), len(dataset) - int(.8 * len(dataset))])
    >>> train_dataloader = DataLoader(train_dataset, batch_size=512)
    >>> test_dataloader = DataLoader(test_dataset, batch_size=512)

    >>> features, labels = next(iter(train_dataloader))
    >>> features[0]
    tensor([[1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
             0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
             0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]])

    >>> labels[0]
    tensor([0.])

File Node Sampler Dataset
================

File node sampler, memory efficient.


.. code-block:: python

    >>> class DeepGNNDataset(Dataset):
    ...     """Cora dataset with file sampler."""
    ...     def __init__(self, data_dir: str, node_types: List[int], feature_meta: List[int]):
    ...         self.g = Client(data_dir, [0, 1])
    ...         self.node_types = np.array(node_types)
    ...         self.feature_meta = feature_meta
    ...         self.count = self.g.node_count(self.node_types)
    ...
    ...     def __len__(self):
    ...         return self.count
    ... 
    ...     def __getitem__(self, idx: int) -> Tuple[Any, Any]:
    ...         return self.g.node_features([idx], np.array([self.feature_meta]), feature_type=np.float32), torch.Tensor([0])


    >>> class FileSampler(Sampler[int]):  # Shouldn't need this really with quick map from torch sampler?
    ...     def __init__(self, filename: str):
    ...         self.filename = filename
    ... 
    ...     def __len__(self) -> int:
    ...         raise NotImplementedError("")
    ... 
    ...     def __iter__(self) -> Iterator[int]:
    ...         with open(self.filename, "r") as file:
    ...             while True:
    ...                 yield int(file.readline())

    >>> dataset = DeepGNNDataset("/tmp/cora", [0, 1, 2], [1, 50])
    >>> train_dataloader = DataLoader(dataset, sampler=FileSampler("/tmp/cora/train.nodes"), batch_size=512)
    >>> test_dataloader = DataLoader(dataset, sampler=FileSampler("/tmp/cora/test.nodes"), batch_size=512)


    >>> features, labels = next(iter(train_dataloader))
    >>> features[0]
    tensor([[3., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
             0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
             0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]])

    >>> labels[0]
    tensor([0.])

Weighted Sampler with Split on Train / Test nodes
================

For using diff types as diff modes

.. code-block:: python
    >>> class DeepGNNDataset(Dataset):
    ...     """Cora dataset with file sampler."""
    ...     def __init__(self, data_dir: str, node_types: List[int], feature_meta: List[int]):
    ...         self.g = Client(data_dir, [0])
    ...         self.node_types = np.array(node_types)
    ...         self.feature_meta = feature_meta
    ...         self.count = self.g.node_count(self.node_types)
    ... 
    ...     def __len__(self):
    ...         return self.count
    ... 
    ...     def __getitem__(self, idx: int) -> Tuple[Any, Any]:
    ...         return self.g.node_features([idx], np.array([self.feature_meta]), feature_type=np.float32), torch.Tensor([0])


    >>> class WeightedSampler(Sampler[int]):  # Shouldn't need this really with quick map from torch sampler?
    ...     def __init__(self, graph: Client, node_types: List[int]):
    ...         self.g = graph
    ...         self.node_types = np.array(node_types)
    ...         self.count = self.g.node_count(self.node_types)
    ... 
    ...     def __len__(self):
    ...         return self.count
    ... 
    ...     def __iter__(self) -> Iterator[int]:
    ...         for _ in range(len(self)):
    ...             yield self.g.sample_nodes(1, self.node_types, SamplingStrategy.Weighted)[0]

    >>> dataset = DeepGNNDataset("/tmp/cora", [0, 1, 2], [1, 50])
    >>> train_dataloader = DataLoader(dataset, sampler=WeightedSampler(dataset.g, node_types=[0]), batch_size=512)
    >>> test_dataloader = DataLoader(dataset, sampler=WeightedSampler(dataset.g, node_types=[1]), batch_size=512)

    >>> features, labels = next(iter(train_dataloader))
    >>> features[0]
    tensor([[4., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
             0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
             0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]])

    >>> labels[0]
    tensor([0.])


Batched Sampler
================

If you have some code with inconsistent outputs where the dataloader collate function
will throw shape errors, here is a work around to maintain high batch sizes.

.. code-block:: python

    >>> class BatchedSampler:
    ...     def __init__(self, sampler, batch_size):
    ...         self.sampler = sampler
    ...         self.batch_size = batch_size
    ...
    ...     def __len__(self):
    ...         return len(self.sampler) // self.batch_size
    ...
    ...     def __iter__(self) -> Iterator[int]:
    ...         generator = iter(self.sampler)
    ...         x = []
    ...         while True:
    ...             try:
    ...                 for _ in range(self.batch_size):
    ...                     x.append(next(generator))
    ...                 yield np.array(x, dtype=np.int64)
    ...                 x = []
    ...             except Exception:
    ...                 break
    ...         if len(x):
    ...             yield np.array(x, dtype=np.int64)


    >>> dataset = DeepGNNDataset(g.data_dir(), [0, 1, 2], [0, g.FEATURE_DIM], [1, 1], np.float32, np.float32)

    >>> ds = DataLoader(dataset, sampler=BatchedSampler(FileSampler(os.path.join(g.data_dir(), "train.nodes")), 140))
