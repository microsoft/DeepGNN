****************************
Node Classification with GAT
****************************

In this guide we use a pre-built `Graph Attention Network(GAT) <https://arxiv.org/abs/1710.10903>`_ model to classify nodes in the `Cora dataset <https://graphsandnetworks.com/the-cora-dataset/>`_. Readers can expect an understanding of the DeepGNN experiment flow and details on model design.

Cora Dataset
============
The Cora dataset consists of 2708 scientific publications represented as nodes interconnected by 5429 reference links represented as edges. Each paper is described by a binary mask for 1433 pertinent dictionary words and an integer in {0..6} representing its type.
First we download the Cora dataset and convert it to a valid binary representation via our built-in Cora downloader.

.. code-block:: python

	>>> from deepgnn.graph_engine.data.citation import Cora
	>>> Cora("/tmp/cora/")
	<deepgnn.graph_engine.data.citation.Cora object at 0x...>

GAT Model
=========

Using this Graph Attention Network, we can accurately predict which category a specific paper belongs to based on its dictionary and the dictionaries of papers it references.
This model leverages masked self-attentional layers to address the shortcomings of graph convolution based models. By stacking layers in which nodes are able to attend over their neighborhoods features, we enable the model to specify different weights to different nodes in a neighborhood, without requiring any kind of costly matrix operation (such as inversion) or the knowledge of the graph structure up front.

`Paper <https://arxiv.org/abs/1710.10903>`_, `author's code <https://github.com/PetarV-/GAT>`_.

Next we copy the GAT model from `DeepGNN's examples directory <https://github.com/microsoft/DeepGNN/blob/main/examples/pytorch/gat>`_. Pre-built models are kept out of the pip installation because it is rarely possible to inheret and selectively edit a single function of a graph model, instead it is best to copy the entire model and edit as needed.
DeepGNN models typically contain multiple parts:

	1. Query struct and implementation
	2. Model init and forward
	3. Training setup: Dataset, Optimizer, Model creation
	4. Execution

Setup
======

Combined imports from `model.py <https://github.com/microsoft/DeepGNN/blob/main/examples/pytorch/gat/model.py>`_ and `main.py <https://github.com/microsoft/DeepGNN/blob/main/examples/pytorch/gat/main.py>`_.

.. code-block:: python

	>>> from typing import List
	>>> from dataclasses import dataclass
	>>> import argparse
	>>> import numpy as np
	>>> import torch
	>>> import torch.nn as nn
	>>> import torch.nn.functional as F
	>>> import deepgnn.pytorch
	>>> from deepgnn.pytorch.common import Accuracy
	>>> from deepgnn.pytorch.modeling.base_model import BaseModel
	>>> from deepgnn.pytorch.nn.gat_conv import GATConv
	>>> from deepgnn.graph_engine import Graph, graph_ops
	>>> from deepgnn import str2list_int
	>>> from deepgnn.pytorch.common.utils import set_seed
	>>> from deepgnn.pytorch.common.dataset import TorchDeepGNNDataset
	>>> from deepgnn.pytorch.modeling import BaseModel
	>>> from deepgnn.pytorch.training import run_dist
	>>> from deepgnn.graph_engine import FileNodeSampler, GraphEngineBackend

Query
=====
Query is the interface between the model and graph engine. It is used by the trainer to fetch contexts which will be passed as input to the model forward function. Since query is a separate function, the trainer may pre-fetch contexts allowing graph engine operations and model training to occur in parallel.
In the GAT model, query samples neighbors repeatedly `num_hops` times in order to generate a sub-graph. All node and edge features in this sub-graph are pulled and added to the context.

.. code-block:: python

	>>> @dataclass
	... class GATQueryParameter:
	...     neighbor_edge_types: np.array
	...     feature_idx: int
	...     feature_dim: int
	...     label_idx: int
	...     label_dim: int
	...     feature_type: np.dtype = np.float32
	...     label_type: np.dtype = np.float32
	...     num_hops: int = 2
	>>> class GATQuery:
	...     def __init__(self, p: GATQueryParameter):
	...         self.p = p
	...         self.label_meta = np.array([[p.label_idx, p.label_dim]], np.int32)
	...         self.feat_meta = np.array([[p.feature_idx, p.feature_dim]], np.int32)
	...
	...     def query_training(self, graph: Graph, inputs):
	...         nodes, edges, src_idx = graph_ops.sub_graph(
	...             graph,
	...             inputs,
	...             edge_types=self.p.neighbor_edge_types,
	...             num_hops=self.p.num_hops,
	...             self_loop=True,
	...             undirected=True,
	...             return_edges=True,
	...         )
	...         input_mask = np.zeros(nodes.size, np.bool)
	...         input_mask[src_idx] = True
	...
	...         feat = graph.node_features(nodes, self.feat_meta, self.p.feature_type)
	...         label = graph.node_features(nodes, self.label_meta, self.p.label_type)
	...         label = label.astype(np.int32)
	...         edges_value = np.ones(edges.shape[0], np.float32)
	...         edges = np.transpose(edges)
	...         adj_shape = np.array([nodes.size, nodes.size], np.int64)
	...
	...         graph_tensor = (nodes, feat, input_mask, label, edges, edges_value, adj_shape)
	...         return graph_tensor

Model Forward and Init
======================
The model init and forward functions look the same as any other pytorch model, except we base off of `deepgnn.pytorch.modeling.base_model.BaseModel` instead of `torch.nn.Module`. The forward function is expected to return three values: the batch loss, the model predictions for given nodes and corresponding labels.
In the GAT model, forward pass uses two of our built-in `GATConv layers <https://github.com/microsoft/DeepGNN/blob/main/src/python/deepgnn/pytorch/nn/gat_conv.py>`_ and computes the loss via cross entropy.

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
	...         q_param: GATQueryParameter = None,
	...     ):
	...         self.q = GATQuery(q_param)
	...         super().__init__(np.float32, 0, 0, None)
	...         self.num_classes = num_classes
	...
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
	...         self.metric = Accuracy()
	...
	...     def forward(self, inputs):
	...         nodes, feat, mask, labels, edges, edges_value, adj_shape = inputs
	...         nodes = torch.squeeze(nodes)                # [N], N: num of nodes in subgraph
	...         feat = torch.squeeze(feat)                  # [N, F]
	...         mask = torch.squeeze(mask)                  # [N]
	...         labels = torch.squeeze(labels)              # [N]
	...         edges = torch.squeeze(edges)                # [X, 2], X: num of edges in subgraph
	...         edges_value = torch.squeeze(edges_value)    # [X]
	...         adj_shape = torch.squeeze(adj_shape)        # [2]
	...
	...         sp_adj = torch.sparse_coo_tensor(edges, edges_value, adj_shape.tolist())
	...         h_1 = self.input_layer(feat, sp_adj)
	...         scores = self.out_layer(h_1, sp_adj)
	...
	...         labels = labels.type(torch.int64)
	...         labels = labels[mask]  # [batch_size]
	...         scores = scores[mask]  # [batch_size]
	...         pred = scores.argmax(dim=1)
	...         loss = self.xent(scores, labels)
	...         return loss, pred, labels

Model Init
==========
We need to implement `create_model` and `create_optimizer` functions to allow distributed workers initialize model and optimizer.

.. code-block:: python

	>>> def create_model(args: argparse.Namespace):
	...     if args.seed:
	...         set_seed(args.seed)
	...
	...     p = GATQueryParameter(
	...         neighbor_edge_types=np.array([args.neighbor_edge_types], np.int32),
	...         feature_idx=args.feature_idx,
	...         feature_dim=args.feature_dim,
	...         label_idx=args.label_idx,
	...         label_dim=args.label_dim,
	...     )
	...
	...     return GAT(
	...         in_dim=args.feature_dim,
	...         head_num=args.head_num,
	...         hidden_dim=args.hidden_dim,
	...         num_classes=args.num_classes,
	...         ffd_drop=args.ffd_drop,
	...         attn_drop=args.attn_drop,
	...         q_param=p,
	...     )
	>>> def create_optimizer(args: argparse.Namespace, model: BaseModel, world_size: int):
	...     return torch.optim.Adam(
	...         filter(lambda p: p.requires_grad, model.parameters()),
	...         lr=args.learning_rate * world_size,
	...         weight_decay=0.0005,
	...     )

Dataset
=======
`create_dataset` function allows parameterization torch of the training data used by workers.
Notably we use the `FileNodeSampler` here which loads `sample_files` and generates samples from them, otherwise in our `link prediction example <link_pred.html>`_ we use `GEEdgeSampler` which uses the backend to generate samples.

.. code-block:: python

	>>> def create_dataset(
	...     args: argparse.Namespace,
	...     model: BaseModel,
	...     rank: int = 0,
	...     world_size: int = 1,
	...     backend: GraphEngineBackend = None,
	... ):
	...     return TorchDeepGNNDataset(
	...         sampler_class=FileNodeSampler,
	...         backend=backend,
	...         query_fn=model.q.query_training,
	...         prefetch_queue_size=2,
	...         prefetch_worker_size=2,
	...         sample_files=args.sample_file,
	...         batch_size=args.batch_size,
	...         shuffle=True,
	...         drop_last=True,
	...         worker_index=rank,
	...         num_workers=world_size,
	...     )

Arguments
=========
`init_args` registers any model specific arguments.

.. code-block:: python

	>>> def init_args(parser):
	...     parser.add_argument("--head_num", type=str2list_int, default="8,1", help="the number of attention headers.")
	...     parser.add_argument("--hidden_dim", type=int, default=8, help="hidden layer dimension.")
	...     parser.add_argument("--num_classes", type=int, default=-1, help="number of classes for category")
	...     parser.add_argument("--ffd_drop", type=float, default=0.0, help="feature dropout rate.")
	...     parser.add_argument("--attn_drop", type=float, default=0.0, help="attention layer dropout rate.")
	...     parser.add_argument("--l2_coef", type=float, default=0.0005, help="l2 loss")
	...     parser.add_argument("--neighbor_edge_types", type=str2list_int, default="0", help="Graph Edge for attention encoder.",)
	...     parser.add_argument("--eval_file", default="", type=str, help="")

NOTE Below code block is for jupyter notebooks only.

.. code-block:: python

	>>> try:
	...     init_args_base
	... except NameError:
	...     init_args_base = init_args
	>>> MODEL_DIR = f"~/tmp/gat_{np.random.randint(9999999)}"
	>>> arg_list = [
	...     "--data_dir", "/tmp/cora",
	...     "--mode", "train",
	...     "--trainer", "base",
	...     "--backend", "snark",
	...     "--graph_type", "local",
	...     "--converter", "skip",
	...     "--sample_file", "/tmp/cora/train.nodes",
	...     "--node_type", "0",
	...     "--feature_idx", "0",
	...     "--feature_dim", "1433",
	...     "--label_idx", "1",
	...     "--label_dim", "1",
	...     "--num_classes", "7",
	...     "--batch_size", "140",
	...     "--learning_rate", ".005",
	...     "--num_epochs", "20",
	...     "--log_by_steps", "10",
	...     "--use_per_step_metrics",
	...     "--data_parallel_num", "0",
	...     "--model_dir", MODEL_DIR,
	...     "--metric_dir", MODEL_DIR,
	...     "--save_path", MODEL_DIR,
	... ]
	>>> def init_args_wrap(init_args_base):
	...     def init_args_new(parser):
	...         init_args_base(parser)
	...         parse_args = parser.parse_args
	...         parser.parse_args = lambda: parse_args(arg_list)
	...     return init_args_new
	>>> init_args = init_args_wrap(init_args_base)

Train
=====
Finally we can train the model with `run_dist` function. We expect the loss to decrease with every epoch:

.. code-block:: python

	>>> run_dist(
	...     init_model_fn=create_model,
	...     init_dataset_fn=create_dataset,
	...     init_optimizer_fn=create_optimizer,
	...     init_args_fn=init_args,
	... )
