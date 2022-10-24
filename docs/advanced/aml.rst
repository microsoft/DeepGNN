**********************
Submit Job to Azure ML
**********************

This example demonstrates how to use ray to submit a job to Azure ML using `ray-on-aml <https://github.com/microsoft/ray-on-aml>`.

Setup
============

See docs/torch/node_class.rst for more details. This will work on TF as well.

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
	>>> from deepgnn.graph_engine.data.citation import Cora

	>>> Cora("/tmp/cora/")
	<deepgnn.graph_engine.data.citation.Cora object at 0x...>

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

    >>> def train_func(config: Dict):
    ...     train.torch.enable_reproducibility(seed=0)
    ...
    ...     model = GAT(in_dim=1433, num_classes=7)
    ...     model = train.torch.prepare_model(model)
    ...
    ...     optimizer = torch.optim.Adam(model.parameters(), lr=.005, weight_decay=0.0005)
    ...     optimizer = train.torch.prepare_optimizer(optimizer)
    ...
    ...     loss_fn = nn.CrossEntropyLoss()
    ...
    ...     dataset = ray.data.range(2708, parallelism=1)
    ...     pipe = dataset.window(blocks_per_window=10)
    ...     g = Client("/tmp/cora", [0], delayed_start=True)
    ...     q = GATQuery()
    ...     def transform_batch(batch: list) -> dict:
    ...         return q.query(g, batch)
    ...     pipe = pipe.map_batches(transform_batch)
    ...
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

Ray Connect to AML
==================


.. code-block:: python

    >>> import os
    >>> os.environ["HOME"] = "."


    >>> from azureml.core import Workspace
    >>> from ray_on_aml.core import Ray_On_AML
    >>> ws = Workspace.from_config("config.json")
    >>> #ray_on_aml = Ray_On_AML(ws=ws, compute_cluster="multi-node", maxnode=1) 
    >>> #ray = ray_on_aml.getRay() 
    # may take 7 mintues or longer.Check the AML run under ray_on_aml experiment for cluster status.  

    >>> #ray.init()
    RayContext(...)
    >>> '''trainer = TorchTrainer(
    ...     train_func,
    ...     train_loop_config={},
    ...     run_config=RunConfig(verbose=0),
    ...     scaling_config=ScalingConfig(num_workers=1, use_gpu=False),
    ... )'''
    >>> #result = trainer.fit()

    >>> #ray_on_aml.shutdown()
