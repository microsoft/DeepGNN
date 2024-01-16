*****************************************
Distributed training of a GraphSage model
*****************************************

In this guide we use a a PyG unsupervised graphsage model to classify nodes in the `PPI dataset <https://paperswithcode.com/dataset/ppi>`_. Distributed training is powered by Ray. It is easy to start and synchornize training with it.

PPI Dataset
============
The PPI dataset consists of 20 graphs representing a different human tissue. Positional gene sets are used, motif gene sets and immunological signatures(50 dimensional vector) as features and gene ontology sets as labels (121 in total)
First we download the PPI dataset and convert it to a valid binary representation via our built-in PPI downloader.

.. code-block:: python

    >>> from deepgnn.graph_engine.data.ppi import PPI
    >>> ppi_data = PPI()


GraphSage Model
===============

We'll reuse the :doc:`link_pred` example.

Dataset
=======


.. code-block:: python

    >>> import numpy as np
    >>> import torch
    >>> from torch.utils.data import DataLoader, IterableDataset
    >>> from deepgnn.graph_engine import Graph, SamplingStrategy
    >>> class SageDataset(IterableDataset):
    ...     def __init__(
    ...         self,
    ...         batch_size: int,
    ...         fanout: list,
    ...         graph: Graph,
    ...         feature_dim: int,
    ...         label_dim: int,
    ...         num_nodes: int,
    ...         seed_edge_type: int,
    ...         fanout_edge_types: np.ndarray,
    ...         count: int = 0,
    ...         neg_node_type: list = None,
    ...     ):
    ...         """Initialize graph query."""
    ...         super(SageDataset, self).__init__()
    ...         self.batch_size = batch_size
    ...         self.num_nodes = num_nodes
    ...         self.graph = graph
    ...         self.fanout = fanout
    ...         self.feature_dim = feature_dim
    ...         self.seed_edge_type = seed_edge_type
    ...         self.fanout_edge_types = fanout_edge_types
    ...         self.count = (
    ...             count * batch_size
    ...             if count > 0
    ...             else self.graph.edge_count(self.seed_edge_type)
    ...         )
    ...         self.label_dim = label_dim
    ...         self.neg_node_type = [0] if neg_node_type is None else neg_node_type
    ...
    ...     def __iter__(self):
    ...         return map(
    ...             self.query,
    ...             range(0, self.count, self.batch_size),
    ...         )
    ...
    ...     def _make_edge_index(self, seed: np.array):
    ...         """Build edge index similar to returned from PyG's NeighborSampler."""
    ...         fst_hop = self.graph.sample_neighbors(
    ...             seed,
    ...             self.fanout_edge_types,
    ...             self.fanout[0],
    ...         )
    ...         fst_unique = np.unique(fst_hop[0].ravel())
    ...         snd_hop = self.graph.sample_neighbors(
    ...             fst_unique,
    ...             self.fanout_edge_types,
    ...             self.fanout[1],
    ...         )
    ...
    ...         # Dedupe second hop edges for faster training.
    ...         snd_edges = np.stack(
    ...             [fst_unique.repeat(self.fanout[1]), snd_hop[0].ravel()], axis=1
    ...         )
    ...         snd_edges = np.unique(snd_edges, axis=0)
    ...         edges = np.concatenate(
    ...             [
    ...                 seed.repeat(self.fanout[0]),
    ...                 snd_edges[:, 0],
    ...                 fst_hop[0].ravel(),
    ...                 snd_edges[:, 1],
    ...             ]
    ...         )
    ...
    ...         # np.unique returns sorted elements, but we need to preserve original order
    ...         # to track labels from the seed array. We do it with argsort to get unique elements
    ...         # in the original order and broadcasting to get inverse indices
    ...         unique_nodes, first_occurrence_indices = np.unique(edges, return_index=True)
    ...         sort_order = np.argsort(first_occurrence_indices)
    ...         ordered_unique_nodes = unique_nodes[sort_order]
    ...         broadcasted_comparison = edges[:, None] == ordered_unique_nodes
    ...         inverse_indices = np.argmax(broadcasted_comparison, axis=1)
    ...
    ...         edge_len = len(edges) // 2
    ...         col = inverse_indices[:edge_len]
    ...         row = inverse_indices[edge_len:]
    ...         return ordered_unique_nodes, col, row
    ...
    ...     def query(self, _: int) -> tuple:
    ...         edges = self.graph.sample_edges(
    ...             self.batch_size,
    ...             np.array([self.seed_edge_type], dtype=np.int32),
    ...             strategy=SamplingStrategy.Weighted,
    ...         )
    ...         src = edges[:, 0]
    ...         dst = edges[:, 1]
    ...         num_pos = src.shape[0]
    ...         num_neg = num_pos
    ...         neg_edges = self.graph.sample_nodes(
    ...             2 * num_neg,
    ...             np.array(self.neg_node_type, dtype=np.int32),
    ...             SamplingStrategy.Weighted,
    ...         )[0]
    ...         seed = np.concatenate(
    ...             [src, neg_edges[:num_neg], dst, neg_edges[num_neg:]], axis=0
    ...         )
    ...         edge_label = np.zeros(num_pos + num_neg)
    ...         edge_label[:num_pos] = 1
    ...         seed, inverse_seed = np.unique(seed, return_inverse=True)
    ...         edge_label_index = inverse_seed.reshape((2, -1))
    ...         nodes, cols, rows = self._make_edge_index(seed)
    ...         feats = self.graph.node_features(
    ...             nodes,
    ...             np.array([[1, self.feature_dim], [0, self.label_dim]], dtype=np.int32),
    ...             np.float32,
    ...         )
    ...
    ...         return (
    ...             feats[:, : self.feature_dim],
    ...             cols,
    ...             rows,
    ...             edge_label_index,
    ...             edge_label,
    ...             feats[:, self.feature_dim :],
    ...             nodes,
    ...         )

Experiment configuration
========================
Next step is to configure our experiment
.. code-block:: python

    >>> from deepgnn.graph_engine.snark.distributed import Server, Client as DistributedClient
    >>> config = {
    ...     "data_dir": ppi_data.data_dir(),
    ...     "device": torch.device("cpu"),
    ...     "num_epochs": 2,
    ...     "num_nodes": ppi_data.NUM_NODES,
    ...     "address": "localhost:9999",
    ...     "feature_idx": 1,
    ...     "feature_dim": ppi_data.FEATURE_DIM,
    ...     "label_idx": 0,
    ...     "label_dim": 1,
    ...     "num_classes": ppi_data.NUM_CLASSES,
    ...     "batch_size": 256,
    ...     "fanout": [5, 5],
    ... }
    >>> s = Server(config["address"], config["data_dir"], 0, 1)
    >>> def get_graph():
    ...     return DistributedClient([config["address"]])
    >>> config["get_graph"] = get_graph

Train function
==============
We can now define our training function for a single epoch.

.. code-block:: python

    >>> from torch.nn.functional import binary_cross_entropy_with_logits
    >>> def train_model(model, optimizer, loader):
    ...     model.train()
    ...     total_loss = 0
    ...     for batch in loader:
    ...         node_features, rows, cols, edge_label_index, edge_label, _, _ = [b[0] for b in batch]
    ...         edge_index = torch.stack([cols, rows], dim=0)
    ...         optimizer.zero_grad()
    ...         h = model(node_features, edge_index)
    ...         h_src = h[edge_label_index[0]]
    ...         h_dst = h[edge_label_index[1]]
    ...         pred = (h_src * h_dst).sum(dim=-1)
    ...         loss = binary_cross_entropy_with_logits(pred, edge_label)
    ...         loss.backward()
    ...         optimizer.step()
    ...         total_loss += float(loss) * pred.size(0)
    ...
    ...     return total_loss

Evaluation
==========
To evaluate model we train a sci-kit classifier on top of the embeddings on training predictions and then use that classifier on test dataset. It is charachterized with

.. code-block:: python

	>>> from sklearn.metrics import f1_score
    >>> from sklearn.multioutput import MultiOutputClassifier
    >>> from sklearn.linear_model import SGDClassifier
    >>> def make_predictions(model, loader):
    ...     ns, xs, ys = [], [], []
    ...     for batch in loader:
    ...         node_features, rows, cols, _, _, node_label, nodes = [b[0] for b in batch]
    ...         edge_index = torch.stack([cols, rows], dim=0)
    ...         pred = model(node_features, edge_index)
    ...         xs.append(pred.cpu())
    ...         ys.append(node_label.cpu())
    ...         ns.append(nodes.cpu())
    ...     n = torch.cat(ns, dim=0)
    ...     _, idx, counts = torch.unique(
    ...         n, sorted=True, return_inverse=True, return_counts=True
    ...     )
    ...     _, ind_sorted = torch.sort(idx, stable=True)
    ...     cum_sum = counts.cumsum(0)
    ...     cum_sum = torch.cat((torch.tensor([0]), cum_sum[:-1]))
    ...     first_indicies = ind_sorted[cum_sum]
    ...
    ...     x = torch.cat(xs, dim=0)
    ...     y = torch.cat(ys, dim=0)
    ...     return x[first_indicies], y[first_indicies]

    >>> def eval_model(model, train_dataset, test_dataset):
    ...     model.eval()
    ...     x, y = make_predictions(model, train_dataset)
    ...     x, y = x.detach().numpy(), y.detach().numpy()
    ...     clf = MultiOutputClassifier(SGDClassifier(loss="log_loss", penalty="l2"))
    ...     clf.fit(x, y)
    ...
    ...     train_f1 = f1_score(y, clf.predict(x), average="micro")
    ...     x, y = make_predictions(model, test_dataset)
    ...     test_f1 = f1_score(y.detach().numpy(), clf.predict(x.detach().numpy()), average="micro")
    ...     return train_f1, test_f1

Putting it all together in a train function.

.. code-block:: python

    >>> import ray
    >>> import ray.train as train
    >>> from ray.train.torch import TorchTrainer
    >>> from ray.air import session
    >>> from ray.air.config import ScalingConfig
    >>> from deepgnn import get_logger
    >>> from torch_geometric.nn import GraphSAGE

    >>> def _train_func(config: dict):
    ...     train.torch.enable_reproducibility(seed=session.get_world_rank())
    ...     device = torch.device("cpu")
    ...     model = GraphSAGE(
    ...         config["feature_dim"],
    ...         hidden_channels=64,
    ...         num_layers=2,
    ...         out_channels=64,
    ...     ).to(device)
    ...     model = train.torch.prepare_model(model)
    ...
    ...     optimizer = torch.optim.Adam(
    ...         model.parameters(),
    ...         lr=0.005 * session.get_world_size(),
    ...     )
    ...     train_dataset = SageDataset(
    ...         batch_size=config["batch_size"],
    ...         fanout=config["fanout"],
    ...         graph=config["get_graph"](),
    ...         feature_dim=config["feature_dim"],
    ...         label_dim=config["label_dim"],
    ...         num_nodes=config["num_nodes"],
    ...         seed_edge_type=0,
    ...         fanout_edge_types=np.array([0], np.int32),
    ...         count=10,
    ...         neg_node_type=[0],
    ...     )
    ...
    ...     pred_train_dataset = SageDataset(
    ...         batch_size=config["batch_size"],
    ...         fanout=config["fanout"],
    ...         graph=config["get_graph"](),
    ...         feature_dim=config["feature_dim"],
    ...         label_dim=config["label_dim"],
    ...         num_nodes=config["num_nodes"],
    ...         seed_edge_type=0,
    ...         fanout_edge_types=np.array([0, 1], np.int32),
    ...         count=10,
    ...         neg_node_type=[0, 1],
    ...     )
    ...
    ...     test_dataset = SageDataset(
    ...         batch_size=config["batch_size"],
    ...         fanout=config["fanout"],
    ...         graph=config["get_graph"](),
    ...         feature_dim=config["feature_dim"],
    ...         label_dim=config["label_dim"],
    ...         num_nodes=config["num_nodes"],
    ...         seed_edge_type=1,
    ...         fanout_edge_types=np.array([0, 1], np.int32),
    ...         count=10,
    ...         neg_node_type=[2],
    ...     )
    ...     train_dataloader = DataLoader(train_dataset)
    ...     train_dataloader = train.torch.prepare_data_loader(train_dataloader)
    ...     pred_train_dataloader = DataLoader(pred_train_dataset)
    ...     pred_train_dataloader = train.torch.prepare_data_loader(pred_train_dataloader)
    ...     test_dataloader = DataLoader(test_dataset)
    ...     test_dataloader = train.torch.prepare_data_loader(test_dataloader)
    ...
    ...     test_f1 = 0
    ...     for epoch in range(config["num_epochs"]):
    ...         loss = train_model(
    ...             model,
    ...             optimizer,
    ...             train_dataloader,
    ...         )
    ...         loss /= train_dataset.count
    ...         train_f1, test_f1 = eval_model(model, pred_train_dataloader, test_dataloader)
    ...         session.report({"test_f1": test_f1.item(), "train_f1": test_f1.item()})

    >>> ray.init(num_cpus=3, log_to_driver=False)
    RayContext...
    >>> trainer = TorchTrainer(
    ...     _train_func,
    ...     train_loop_config=config,
    ...     scaling_config=ScalingConfig(num_workers=2),
    ... )
    ...
    >>> res = trainer.fit()
    == Status ==...
    >>> res.metrics["test_f1"]
    0.6...
