*****************************************
Hyperparameter Optimization with Ray Tune
*****************************************

In this guide we build on top of the Ray usage example with a Ray Tune example at the bottom.
The following code block is from `node_class example </torch/node_class.html>`_, see this example for more details.

Cora Dataset
============

.. code-block:: python

    >>> import tempfile
	>>> from deepgnn.graph_engine.data.citation import Cora
    >>> data_dir = tempfile.TemporaryDirectory()
	>>> Cora(data_dir.name)
	<deepgnn.graph_engine.data.citation.Cora object at 0x...>

GAT Model
=========

Setup
======

.. code-block:: python

    >>> from typing import List, Tuple, Any, Dict
    >>> from dataclasses import dataclass, field
    >>> import os
    >>> import numpy as np
    >>> import torch
    >>> import torch.nn as nn
    >>> import torch.nn.functional as F

    >>> import ray
    >>> import ray.train as train
    >>> from ray.train.torch import TorchTrainer
    >>> from ray.air import session
    >>> from ray.air.config import ScalingConfig, RunConfig
    >>> from ray import tune

    >>> import deepgnn.pytorch
    >>> from deepgnn.pytorch.nn.gat_conv import GATConv
    >>> from deepgnn.graph_engine import Graph, graph_ops
    >>> from deepgnn.pytorch.modeling import BaseModel

    >>> from deepgnn.graph_engine.snark.distributed import Server, Client as DistributedClient
    >>> from deepgnn.graph_engine.data.citation import Cora

Query
=====

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
    ...     def query(self, g: DistributedClient, idx: int) -> Dict[Any, np.ndarray]:
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

.. code-block:: python

    >>> class GAT(nn.Module):
    ...     def __init__(
    ...         self,
    ...         in_dim: int,
    ...         head_num: List = [8, 1],
    ...         hidden_dim: int = 8,
    ...         num_classes: int = -1,
    ...         ffd_drop: float = 0.0,
    ...         attn_drop: float = 0.0,
    ...     ):
    ...         super().__init__()
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
    ...         sp_adj = torch.sparse_coo_tensor(edges, torch.ones(edges.shape[1], dtype=torch.float32), (nodes.shape[0], nodes.shape[0]))
    ...         h_1 = self.input_layer(feat, sp_adj)
    ...         scores = self.out_layer(h_1, sp_adj)
    ...
    ...         scores = scores[mask]  # [batch_size]
    ...         return scores


Ray Tune
========

First we define a standard torch training loop using the ray dataset.

.. code-block:: python

    >>> def train_func(config: Dict):
    ...     # Set random seed
    ...     train.torch.enable_reproducibility(seed=session.get_world_rank())
    ...
    ...     # Start server
    ...     address = "localhost:9999"
    ...     s = Server(address, config["data_dir"], 0, 1)
    ...     g = DistributedClient(address)
    ...
    ...     # Initialize the model and wrap it with Ray
    ...     model = GAT(in_dim=1433, num_classes=7)
    ...     model = train.torch.prepare_model(model)
    ...
    ...     # Initialize the optimizer and wrap it with Ray
    ...     optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"], weight_decay=0.0005)
    ...     optimizer = train.torch.prepare_optimizer(optimizer)
    ...
    ...     # Define the loss function
    ...     loss_fn = nn.CrossEntropyLoss()
    ...
    ...     # Ray Dataset
    ...     dataset = ray.data.range(2708).repartition(2708 // config["batch_size"])  # -> Dataset(num_blocks=6, num_rows=2708, schema=<class 'int'>)
    ...     pipe = dataset.window(blocks_per_window=10).repeat(config["n_epochs"])  # -> DatasetPipeline(num_windows=1, num_stages=1)
    ...     q = GATQuery()
    ...     def transform_batch(batch: list) -> dict:
    ...         return q.query(g, batch)  # When we reference the server g in transform, it uses Client instead
    ...     pipe = pipe.map_batches(transform_batch)
    ...
    ...     # Execute the training loop
    ...     model.train()
    ...     for epoch, epoch_pipe in enumerate(pipe.iter_epochs()):
    ...         epoch_pipe = epoch_pipe.random_shuffle_each_window()
    ...         for i, batch in enumerate(epoch_pipe.iter_torch_batches(batch_size=config["batch_size"])):
    ...             scores = model(batch)
    ...             labels = batch["labels"][batch["input_mask"]].flatten()
    ...             loss = loss_fn(scores.type(torch.float32), labels)
    ...             optimizer.zero_grad()
    ...             loss.backward()
    ...             optimizer.step()
    ...
    ...     session.report({"metric": (scores.argmax(1) == labels).float().mean().item()})

Now we define the objective function using this trainer. The objective function will
take a set of parameters from the tuner and return a fitness value.

.. code-block:: python

    >>> ray.init(num_cpus=8)
    RayContext(...)

    >>> def objective(learning_rate, n_epochs):
    ...     trainer = TorchTrainer(
    ...         train_func,
    ...         train_loop_config={
    ...             "batch_size": 2708,
    ...             "data_dir": data_dir.name,
    ...             "sample_filename": "train.nodes",
    ...             "n_epochs": n_epochs,
    ...             "learning_rate": learning_rate,
    ...         },
    ...         run_config=RunConfig(verbose=0),
    ...         scaling_config=ScalingConfig(num_workers=1, use_gpu=False),
    ...     )
    ...     result = trainer.fit()
    ...     return result

This training function wraps this objective function for use in the tuner.

.. code-block:: python

    >>> def training_function(config):
    ...    results = objective(config["learning_rate"], config["n_epochs"])
    ...    tune.report(accuracy=results.metrics["metric"])

Finally we define and make use of the tuner. We use the hyperparameters `learning_rate` and
`n_epochs`, with one training iteration per configuration.

See the `Ray Tune guides, here<https://docs.ray.io/en/latest/tune/tutorials/overview.html>`_.

.. code-block:: python

    >>> tuner = tune.Tuner(
    ...     training_function,
    ...     param_space={
    ...         "learning_rate": tune.grid_search([.05, .005, .0005]),
    ...         "n_epochs": tune.choice([2, 4])
    ...     },
    ...     tune_config=tune.TuneConfig(num_samples=1),
    ...     run_config=RunConfig(
    ...         stop={"training_iteration": 1},
    ...         verbose=0,
    ...     ),
    ... )
    >>> analysis = tuner.fit()

    >>> analysis.get_best_result(metric="accuracy", mode="max")
    Result(metrics={'accuracy': ..., 'experiment_tag': '..._learning_rate=...,n_epochs=...'}, ...)

    >>> analysis.get_dataframe()
       accuracy  ...
    0  0.3...
    1  0.3...
    [3 rows x 21 columns]
