****************************
Node Classification with GAT
****************************

In this guide we use a pre-built `Graph Attention Network(GAT) <https://arxiv.org/abs/1710.10903>`_ model
to classify nodes in the `Cora dataset <https://graphsandnetworks.com/the-cora-dataset/>`_. Readers can
expect an understanding of the DeepGNN experiment flow and details on model design.

Cora Dataset
============
The Cora dataset consists of 2708 scientific publications represented as nodes interconnected by 5429
reference links represented as edges. Each paper is described by a binary mask for 1433 pertinent
dictionary words and an integer in {0..6} representing its type.
First we download the Cora dataset and convert it to a valid binary representation via our built-in Cora downloader.

.. code-block:: python

    >>> from deepgnn.graph_engine.data.citation import Cora
    >>> Cora("/tmp/cora/")
    <deepgnn.graph_engine.data.citation.Cora object at 0x...>

GAT Model
=========

Using this Graph Attention Network, we can accurately predict which category a specific paper belongs to
based on its dictionary and the dictionaries of papers it references.
This model leverages masked self-attentional layers to address the shortcomings of graph convolution
based models. By stacking layers in which nodes are able to attend over their neighborhoods features,
we enable the model to specify different weights to different nodes in a neighborhood, without requiring
any kind of costly matrix operation (such as inversion) or the knowledge of the graph structure up front.

`Paper <https://arxiv.org/abs/1710.10903>`_, `author's code <https://github.com/PetarV-/GAT>`_.

Next we copy the GAT model from `DeepGNN's examples directory <https://github.com/microsoft/DeepGNN/blob/main/examples/pytorch/gat>`_.
Pre-built models are kept out of the pip installation because it is rarely possible to inheret and selectively
edit a single function of a graph model, instead it is best to copy the entire model and edit as needed.
DeepGNN models typically contain multiple parts:

    1. Query struct and implementation
    2. Model init and forward
    3. Training setup: Dataset, Optimizer, Model creation
    4. Execution

Setup
======

.. code-block:: python

    >>> from typing import List, Tuple, Any, Dict
    >>> from dataclasses import dataclass, field
    >>> import tempfile
    >>> import numpy as np
    >>> import torch
    >>> import torch.nn as nn
    >>> import torch.nn.functional as F

    >>> import ray
    >>> import ray.train as train
    >>> from ray.train.torch import TorchTrainer
    >>> from ray.air import session
    >>> from ray.air.config import ScalingConfig, RunConfig

    >>> import deepgnn.pytorch
    >>> from deepgnn.pytorch.nn.gat_conv import GATConv
    >>> from deepgnn.graph_engine import Graph, graph_ops
    >>> from deepgnn.graph_engine.snark.local import Client
    >>> from deepgnn.pytorch.modeling import BaseModel

Query
=====

The Query function takes a set of node or edge ids and returns the 
neighbor and feature information about these items that will be given to the model forward function.
Query is kept in a separate object from the model because it will be run in
a ray data pipeline in the background, not in the main thread.
Query returns can have several different return types, see `ray batch format docs <https://docs.ray.io/en/latest/data/api/dataset.html#ray.data.Dataset.map_batches:~:text=batch_format%20%E2%80%93%20Specify%20%E2%80%9Cnative,Default%20is%20%E2%80%9Cnative%E2%80%9D>`.
In this example we use a dictionary
with all shapes starting with dimension 0 having size 1.

In the GAT model, query samples neighbors repeatedly `num_hops` times in
order to generate a sub-graph.
All node and edge features in this sub-graph are pulled and added to the context.

.. code-block:: python

    >>> @dataclass
    ... class GATQuery:
    ...     feature_meta: list = field(default_factory=lambda: np.array([[0, 1433]]))
    ...     label_meta: list = field(default_factory=lambda: np.array([[1, 1]]))
    ...     feature_type: np.dtype = np.float32
    ...     label_type: np.dtype = np.float32
    ...     neighbor_edge_types: list = field(default_factory=lambda: [0])
    ...     num_hops: int = 2
    ...
    ...     def query(self, g: Client, idx: int) -> Dict[Any, np.ndarray]:
    ...         """Query used to generate data for training."""
    ...         if isinstance(idx, (int, float)):
    ...             idx = [idx]
    ...         inputs = np.array(idx, np.int64)
    ...         nodes, edges, src_idx = graph_ops.sub_graph(
    ...             g,
    ...             inputs,
    ...             edge_types=np.array(self.neighbor_edge_types, np.int64),
    ...             num_hops=self.num_hops,
    ...             self_loop=True,
    ...             undirected=True,
    ...             return_edges=True,
    ...         )
    ...         input_mask = np.zeros(nodes.size, np.bool)
    ...         input_mask[src_idx] = True
    ...
    ...         feat = g.node_features(nodes, self.feature_meta, self.feature_type)
    ...         label = g.node_features(nodes, self.label_meta, self.label_type).astype(np.int64)
    ...         return {"nodes": np.expand_dims(nodes, 0), "feat": np.expand_dims(feat, 0), "labels": np.expand_dims(label, 0), "input_mask": np.expand_dims(input_mask, 0), "edges": np.expand_dims(edges, 0)}


Model Forward and Init
======================
The model init and forward functions look the same as any other pytorch model,
except we base off of `deepgnn.pytorch.modeling.base_model.BaseModel` instead of `torch.nn.Module`.
The forward function is expected to return the model scores for the given nodes.
In the GAT model, forward pass uses two of our built-in
`GATConv layers <https://github.com/microsoft/DeepGNN/blob/main/src/python/deepgnn/pytorch/nn/gat_conv.py>`_
and computes the loss via cross entropy.

.. code-block:: python

    >>> class GAT(BaseModel):
    ...     def __init__(
    ...         self,
    ...         in_dim: int,
    ...         head_num: List = [8, 1],
    ...         hidden_dim: int = 8,
    ...         num_classes: int = -1,
    ...         ffd_drop: float = 0.0,
    ...         attn_drop: float = 0.0,
    ...     ):
    ...         super().__init__(np.float32, 0, 0, None)
    ...         self.num_classes = num_classes
    ...         self.out_dim = num_classes
    ...
    ...         self.input_layer = GATConv(
    ...             in_dim=in_dim,
    ...             attn_heads=head_num[0],
    ...             out_dim=hidden_dim,
    ...             act=F.elu,
    ...             in_drop=ffd_drop,
    ...             coef_drop=attn_drop,
    ...             attn_aggregate="concat",
    ...         )
    ...         layer0_output_dim = head_num[0] * hidden_dim
    ...         assert len(head_num) == 2
    ...         self.out_layer = GATConv(
    ...             in_dim=layer0_output_dim,
    ...             attn_heads=head_num[1],
    ...             out_dim=self.out_dim,
    ...             act=None,
    ...             in_drop=ffd_drop,
    ...             coef_drop=attn_drop,
    ...             attn_aggregate="average",
    ...         )
    ...
    ...     def forward(self, context: Dict[Any, np.ndarray]):
    ...         nodes = torch.squeeze(context["nodes"])                # [N], N: num of nodes in subgraph
    ...         feat = torch.squeeze(context["feat"])                  # [N, F]
    ...         mask = torch.squeeze(context["input_mask"])            # [N]
    ...         labels = torch.squeeze(context["labels"])              # [N]
    ...         edges = torch.squeeze(context["edges"].reshape((-1, 2)))                # [X, 2], X: num of edges in subgraph
    ...
    ...         edges = np.transpose(edges)
    ...
    ...         # TODO This is not stable, when doing batch_size < graph size ends up with size < index values. use torch.unique to remap edges
    ...         sp_adj = torch.sparse_coo_tensor(edges, torch.ones(edges.shape[1], dtype=torch.float32), (nodes.shape[0], nodes.shape[0]))
    ...         h_1 = self.input_layer(feat, sp_adj)
    ...         scores = self.out_layer(h_1, sp_adj)
    ...
    ...         scores = scores[mask]  # [batch_size]
    ...         return scores


Train
=====

Here we define our training function.
In the setup part we do 2 notable things things,

* Wrap the model and optimizer with train.torch.prepare_model/optimizer for Ray multi worker usage.

* Initialize the ray dataset, see more details in `docs/graph_engine/dataset.rst`.

Then we define a standard torch training loop using the ray dataset, with no changes to model or optimizer usage.

.. code-block:: python

    >>> def train_func(config: Dict):
    ...     # Set random seed
    ...     train.torch.enable_reproducibility(seed=0)
    ...
    ...     # Initialize the model and wrap it with Ray
    ...     model = GAT(in_dim=1433, num_classes=7)
    ...     model = train.torch.prepare_model(model)
    ...
    ...     # Initialize the optimizer and wrap it with Ray
    ...     optimizer = torch.optim.Adam(model.parameters(), lr=.005, weight_decay=0.0005)
    ...     optimizer = train.torch.prepare_optimizer(optimizer)
    ...
    ...     # Define the loss function
    ...     loss_fn = nn.CrossEntropyLoss()
    ...
    ...     # Ray Dataset
    ...     dataset = ray.data.range(2708, parallelism=1)  # -> Dataset(num_blocks=1, num_rows=140, schema=<class 'int'>)
    ...     pipe = dataset.window(blocks_per_window=10)  # -> DatasetPipeline(num_windows=1, num_stages=1)
    ...     g = Client("/tmp/cora", [0], delayed_start=True)
    ...     q = GATQuery()
    ...     def transform_batch(batch: list) -> dict:
    ...         return q.query(g, batch)
    ...     pipe = pipe.map_batches(transform_batch)
    ...
    ...     # Execute the training loop
    ...     model.train()
    ...     for epoch, epoch_pipe in enumerate(pipe.repeat(1).iter_epochs()):
    ...         for i, batch in enumerate(epoch_pipe.random_shuffle_each_window().iter_torch_batches(batch_size=2708)):
    ...             scores = model(batch)
    ...             labels = batch["labels"][batch["input_mask"]].flatten()
    ...             loss = loss_fn(scores.type(torch.float32), labels)
    ...             optimizer.zero_grad()
    ...             loss.backward()
    ...             optimizer.step()
    ...
    ...             session.report({"metric": (scores.argmax(1) == labels).sum(), "loss": loss.item()})

In this step we start the training job.
First we start a local ray cluster with `ray.init() <https://docs.ray.io/en/latest/ray-core/package-ref.html#ray-init>`.
Next we initialize a `TorchTrainer <https://docs.ray.io/en/latest/ray-air/package-ref.html#pytorch>`
object to wrap our training loop. This takes parameters that go to the training loop and parameters
to define number workers and cpus/gpus used.
Finally we call trainer.fit() to execute the training loop.

.. code-block:: python

    >>> ray.init()
    RayContext(...)
    >>> trainer = TorchTrainer(
    ...     train_func,
    ...     train_loop_config={},
    ...     run_config=RunConfig(verbose=0),
    ...     scaling_config=ScalingConfig(num_workers=1, use_gpu=False),
    ... )
    >>> result = trainer.fit()
