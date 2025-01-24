****************************
Node Classification with GAT
****************************

In this guide we use a pre-built `Graph Attention Network(GAT) <https://arxiv.org/abs/1710.10903>`_ model to classify nodes in the `Cora dataset <https://graphsandnetworks.com/the-cora-dataset/>`_. Readers can expect an understanding of the DeepGNN experiment flow and details on model design.

Cora Dataset
============
The Cora dataset consists of 2708 scientific publications represented as nodes interconnected by 5429 reference links represented as edges. Each paper is described by a binary mask for 1433 pertinent dictionary words and an integer in {0..6} representing its type.
First we download the Cora dataset and convert it to a valid binary representation via our built-in Cora downloader.

.. code-block:: python

    >>> from deepgnn.graph_engine.data.cora import Cora
    >>> graph = CoraFull()

GAT Model
=========

Using this Graph Attention Network, we can accurately predict which category a specific paper belongs to based on its dictionary and the dictionaries of papers it references.
This model leverages masked self-attentional layers to address the shortcomings of graph convolution based models. By stacking layers in which nodes are able to attend over their neighborhoods features, we enable the model to specify different weights to different nodes in a neighborhood, without requiring any kind of costly matrix operation (such as inversion) or the knowledge of the graph structure up front.

`Paper <https://arxiv.org/abs/1710.10903>`_, `author's code <https://github.com/PetarV-/GAT>`_.

Next we'll create the GAT model. DeepGNN models typically contain multiple parts:

    1. Define query function to create a Dataset.
    2. Model init and forward.
    3. Training and evaluation.

Query
=====
Query is the interface between the model and graph engine. It is used by the trainer to fetch contexts which will be passed as input to the model forward function. Since query is a separate function, the trainer may pre-fetch contexts allowing graph engine operations and model training to occur in parallel.
In the GAT model, query samples neighbors repeatedly `num_hops` times in order to generate a sub-graph. All node and edge features in this sub-graph are pulled and added to the context.

.. code-block:: python

    >>> import numpy as np
    >>> from torch.utils.data import DataLoader, IterableDataset
    >>> from deepgnn.graph_engine import Graph, graph_ops
    >>> class GATDataset(IterableDataset):
    ...     def __init__(self, inputs: np.array, graph: Graph, batch_size:int = 140):
    ...         super(GATDataset, self).__init__()
    ...         self.graph = graph
    ...         self.inputs = inputs
    ...         self.batch_size = batch_size
    ...
    ...     def __iter__(self):
    ...         np.random.shuffle(self.inputs)
    ...         batches = self.inputs
    ...         last_elements = len(self.inputs) % self.batch_size
    ...         if last_elements > 0:
    ...             batches = batches[:-last_elements]
    ...         batches = batches.reshape(-1, self.batch_size)
    ...         return map(self.query, batches)
    ...
    ...     def query(self, inputs: np.ndarray) -> tuple:
    ...         nodes, edges, src_idx = graph_ops.sub_graph(
    ...             self.graph,
    ...             inputs,
    ...             edge_types=np.array([0], dtype=np.int32),
    ...             num_hops=2,
    ...             self_loop=True,
    ...             undirected=True,
    ...             return_edges=True,
    ...         )
    ...         input_mask = np.zeros(nodes.size, np.bool_)
    ...         input_mask[src_idx] = True
    ...         feat = self.graph.node_features(
    ...             nodes, np.array([[0, 1433]], dtype=np.int32), np.float32
    ...         )
    ...         label = self.graph.node_features(
    ...             nodes, np.array([[1, 1]], dtype=np.int32), np.float32
    ...         ).astype(np.int32)
    ...         edges = np.transpose(edges)
    ...
    ...         return (nodes, feat, edges, input_mask, label)

Model Forward and Init
======================
The model init and forward functions look the same as any other pytorch model, except we base off of `deepgnn.pytorch.modeling.base_model.BaseModel` instead of `torch.nn.Module`. The forward function is expected to return three values: the batch loss, the model predictions for given nodes and corresponding labels.
In the GAT model, forward pass uses two of our built-in `GATConv layers <https://github.com/microsoft/DeepGNN/blob/main/src/python/deepgnn/pytorch/nn/gat_conv.py>`_ and computes the loss via cross entropy.

.. code-block:: python

    >>> import torch.nn as nn
    >>> import torch.nn.functional as F
    >>> import torch
    >>> from torch_geometric.nn import GATConv
    >>> class GAT(nn.Module):
    ...    """GAT model."""
    ...
    ...    def __init__(
    ...        self,
    ...        in_dim: int,
    ...        head_num: list = [8, 1],
    ...        hidden_dim: int = 8,
    ...        num_classes: int = -1,
    ...    ):
    ...        """Initialize model."""
    ...        super(GAT, self).__init__()
    ...        self.num_classes = num_classes
    ...        self.out_dim = num_classes
    ...        self.xent = nn.CrossEntropyLoss()
    ...        self.conv1 = GATConv(
    ...            in_channels=in_dim,
    ...            out_channels=hidden_dim,
    ...            heads=head_num[0],
    ...            dropout=0.6,
    ...        )
    ...        layer0_output_dim = head_num[0] * hidden_dim
    ...        self.conv2 = GATConv(
    ...            in_channels=layer0_output_dim,
    ...            out_channels=self.out_dim,
    ...            heads=1,
    ...            dropout=0.6,
    ...            concat=False,
    ...        )
    ...
    ...    def forward(self, inputs: tuple):
    ...        """Calculate loss, make predictions and fetch labels."""
    ...        nodes, feat, edge_index, mask, label = inputs
    ...        nodes = torch.squeeze(nodes.to(torch.int32))  # [N]
    ...        feat = torch.squeeze(feat.to(torch.float32))  # [N, F]
    ...        edge_index = torch.squeeze(edge_index.to(torch.int32))  # [2, X]
    ...        mask = torch.squeeze(mask.to(torch.bool))  # [N]
    ...        labels = torch.squeeze(label.to(torch.int64))  # [N]
    ...
    ...        x = feat
    ...        x = F.dropout(x, p=0.6, training=self.training)
    ...        x = F.elu(self.conv1(x, edge_index))
    ...        x = F.dropout(x, p=0.6, training=self.training)
    ...        scores = self.conv2(x, edge_index)
    ...        labels = labels[mask]  # [batch_size]
    ...        scores = scores[mask]  # [batch_size]
    ...        pred = scores.argmax(dim=1)
    ...        loss = self.xent(scores, labels)
    ...
    ...        return loss, pred, labels

Model Init
==========
We can now create model

.. code-block:: python

    >>> model = GAT(
    ...     in_dim=1433,
    ...     head_num=[8, 1],
    ...     hidden_dim=8,
    ...     num_classes=7, # TODO: extract from cora
    ... )

    >>> optimizer = torch.optim.Adam(
    ...    filter(lambda p: p.requires_grad, model.parameters()),
    ...    lr=0.005,
    ...    weight_decay=0.0005,
    ... )

Dataset
=======
We can now create graph and training dataset.

.. code-block:: python

	>>> import os.path as osp
    >>> train_np = np.loadtxt(osp.join(graph.data_dir(), "test.nodes"), dtype=np.int64)
    >>> train_dataloader = DataLoader(GATDataset(train_np, graph, len(train_np)))


Cora comes with predetermined set of test nodes. We'll use it for evaluation dataset.

.. code-block:: python

    >>> eval_batch = np.loadtxt(osp.join(graph.data_dir(), "test.nodes"), dtype=np.int64)
    >>> eval_dataloader = DataLoader(GATDataset(eval_batch, graph, len(eval_batch)))

Training
=========
Everything is ready to start training. We can get good results from training over 500 epochs. One epoch in this example is iteration through entire training dataset of 140 nodes.

.. code-block:: python

    >>> model.train()
    GAT(
      (xent): CrossEntropyLoss()
      (conv1): GATConv(1433, 8, heads=8)
      (conv2): GATConv(64, 7, heads=1)
    )
    >>> for _ in range(100):
    ...    for batch in train_dataloader:
    ...        loss, _, _ = model(batch)
    ...        optimizer.zero_grad()
    ...        loss.backward()
    ...        optimizer.step()

Evaluaion
=========

We'll use accuracy metric to evaluate performance of our model.

.. code-block:: python

    >>> from sklearn.metrics import accuracy_score
    >>> model.eval()
    GAT(
      (xent): CrossEntropyLoss()
      (conv1): GATConv(1433, 8, heads=8)
      (conv2): GATConv(64, 7, heads=1)
    )
    >>> eval_tensor = next(iter(eval_dataloader))
    >>> _, score, label = model(eval_tensor)
    >>> accuracy = torch.tensor(
    ...     accuracy_score(y_true=label.cpu(), y_pred=score.detach().cpu().numpy())
    ... )
    >>> accuracy.item()
    0.8...
