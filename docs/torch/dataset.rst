****************************
Torch Dataset Usage
****************************

In this guide we create a few Torch Datasets with various sampling strategies then we go through ways of setting a train test split.

Generate Dataset
================

.. code-block:: python

    >>> import numpy as np
    >>> import torch
    >>> from torch.utils.data import Dataset, DataLoader, Sampler

    >>> from deepgnn.graph_engine import SamplingStrategy
    >>> from deepgnn.graph_engine.snark.local import Client

    >>> from deepgnn.graph_engine.data.citation import Cora
    >>> Cora("/tmp/cora")  # Generate Cora dataset (Train: 140, Valid: 500, Test: 1000)
    <deepgnn.graph_engine.data.citation.Cora object at 0x...>

Simple Cora Dataset
================

.. code-block:: python

    >>> class DeepGNNDataset(Dataset):
    ...     """Cora dataset with base torch sampler."""
    ...     def __init__(self, filename, node_types):
    ...         self.g = Client(filename, [0, 1])
    ...         self.node_types = np.array(node_types)
    ...         self.count = self.g.node_count(self.node_types)
    ... 
    ...     def __len__(self):
    ...         return self.count
    ... 
    ...     def __getitem__(self, sampler_idx):
    ...         if isinstance(sampler_idx, (int, float)):
    ...             sampler_idx = [sampler_idx]
    ...         idx = sampler_idx# self.g.type_remap(sampler_idx)  # TODO
    ...         return self.g.node_features(idx, np.array([[feature_idx, feature_dim]]), feature_type=np.float32), torch.Tensor([0])

    >>> dataset = DeepGNNDataset("/tmp/cora", [0])
    >>> train_dataset, test_dataset = torch.utils.data.random_split(dataset, [int(.8 * len(dataset)), len(dataset) - int(.8 * len(dataset))])
    >>> train_dataloader = DataLoader(train_dataset, batch_size=512)
    >>> test_dataloader = DataLoader(test_dataset, batch_size=512)


File Node Sampler Dataset
================

File node sampler, memory efficient.


.. code-block:: python

    >>> class DeepGNNDataset(Dataset):
    ...     """Cora dataset with file sampler."""
    ...     def __init__(self, filename, node_types):
    ...         self.g = Client(filename, [0, 1])
    ...         self.node_types = np.array(node_types)
    ...         self.count = self.g.node_count(self.node_types)
    ...
    ...     def __len__(self):
    ...         return self.count
    ... 
    ...     def __getitem__(self, sampler_idx):
    ...         if isinstance(sampler_idx, (int, float)):
    ...             sampler_idx = [sampler_idx]
    ...         idx = sampler_idx
    ...         return self.g.node_features(idx, np.array([[feature_idx, feature_dim]]), feature_type=np.float32), torch.Tensor([0])


    >>> class FileSampler(Sampler[int]):  # Shouldn't need this really with quick map from torch sampler?
    ...     def __init__(self, filename):
    ...         self.filename = filename
    ... 
    ...     def __len__(self) -> int:
    ...         raise NotImplementedError("")
    ... 
    ...     def __iter__(self):
    ...         with open(self.filename, "r") as file:
    ...             for line in file.readlines():
    ...                 yield int(line)

    >>> dataset = DeepGNNDataset("/tmp/cora", [0])
    >>> train_dataloader = DataLoader(dataset, sampler=FileSampler("/tmp/cora/train.nodes"), batch_size=512)
    >>> test_dataloader = DataLoader(dataset, sampler=FileSampler("/tmp/cora/test.nodes"), batch_size=512)

Weighted Sampler with Split on Train / Test nodes
================

For using diff types as diff modes

.. code-block:: python
    >>> class DeepGNNDataset(Dataset):
    ...     """Cora dataset with file sampler."""
    ...     def __init__(self, filename, node_types):
    ...         self.g = Client(filename, [0])
    ...         self.node_types = np.array(node_types)
    ...         self.count = self.g.node_count(self.node_types)
    ... 
    ...     def __len__(self):
    ...         return self.count
    ... 
    ...     def __getitem__(self, sampler_idx):
    ...         if isinstance(sampler_idx, (int, float)):
    ...             sampler_idx = [sampler_idx]
    ...         idx = sampler_idx
    ...         return self.g.node_features(idx, np.array([[feature_idx, feature_dim]]), feature_type=np.float32), torch.Tensor([0])


    >>> class WeightedSampler(Sampler[int]):  # Shouldn't need this really with quick map from torch sampler?
    ...     def __init__(self, graph, node_types):
    ...         self.g = graph
    ...         self.node_types = np.array(node_types)
    ...         self.count = self.g.node_count(self.node_types)
    ... 
    ...     def __len__(self):
    ...         return self.count
    ... 
    ...     def __iter__(self):
    ...         for _ in range(len(self)):
    ...             yield self.g.sample_nodes(1, self.node_types, SamplingStrategy.Weighted)[0]

    >>> dataset = DeepGNNDataset("/tmp/cora", [0])
    >>> train_dataloader = DataLoader(dataset, sampler=WeightedSampler(dataset.g, node_types=[0]), batch_size=512)
    >>> test_dataloader = DataLoader(dataset, sampler=WeightedSampler(dataset.g, node_types=[1]), batch_size=512)
