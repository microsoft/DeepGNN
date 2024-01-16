*******************************
GraphSage Link Prediction Model
*******************************

We'll take you step-by-step through the process of creating a link prediction model using GraphSage.
This model will be trained on the `Collaborative Filtering (Collab) dataset <https://ogb.stanford.edu/docs/linkprop/#ogbl-collab>`_,
which is provided by Stanford's OGB. For a broader understanding of the entire DeepGNN training pipeline, feel free to consult the `overview guide <quickstart.html>`_.


Downloading the dataset
=======================

Before diving into the model, we need to download the datasetthat will serve as the foundation for training and evaluation. For this guide,
we've chosen the Collab dataset for its suitability for link prediction tasks. The following code fetches the dataset from a designated URL
and unzips its content to a temporary directory.


.. code-block:: python

	>>> import io
	>>> import requests
	>>> import os.path as osp
	>>> import tempfile
	>>> import zipfile
	>>> url = "https://deepgraphpub.blob.core.windows.net/public/testdata/collab.zip"
	>>> response = requests.get(url)
	>>> temp_dir = tempfile.mkdtemp()
	>>> zip_file_like = io.BytesIO(response.content)
	>>> with zipfile.ZipFile(zip_file_like) as zip_ref:
	... 	zip_ref.extractall(temp_dir)

The dataset must be adapted into a format that allows for convenient data manipulation and model training. To do this, we'll convert it to
`ogb.LinkPropPredDataset`, enabling us to isolate edge data and establish train/validation/test splits.

.. code-block:: python

	>>> meta_dict = {
	...     'dir_path': osp.join(temp_dir, "collab"),
	...     'eval metric':'hits@50',
	...     'task type':'link prediction',
	...     'download_name':'collab',
	...     'version':'1',
	...     'url': url,
	...     'add_inverse_edge':'True',
	...     'has_node_attr':'True',
	...     'has_edge_attr':'False',
	...     'split':'time',
	...     'additional node files':'None',
	...     'additional edge files':'edge_weight,edge_year',
	...     'is hetero':'False',
	...     'binary':'False',
	... }
	>>> from ogb.linkproppred import LinkPropPredDataset
	>>> dataset = LinkPropPredDataset(name = 'ogbl-collab', meta_dict=meta_dict)
	Loading necessary files...

Now we can create a binary graph data for DeepGNN to sample neighbors effectively.
We need to sort the edges by source node, then by edge year and finally by destination node before feeding to a binary writer.

.. code-block:: python

	>>> import numpy as np
	>>> edges = np.stack([dataset.graph['edge_index'][0], dataset.graph['edge_index'][1], dataset.graph['edge_year'][:,0], dataset.graph['edge_weight'][:,0]], axis=1)
	>>> sorted_indices = np.lexsort((edges[:, 1], edges[:, 2], edges[:, 0]))
	>>> edges = edges[sorted_indices]
	>>> def edge_iterator(edge_arr: np.array, node_features: np.array):
	...    curr_src = -1 # mark a new source node in the edge list to record node features
	...    for row in edge_arr:
	...        node_id = row[0]
	...        if node_id != curr_src:
	...            yield node_id, -1, 0, 1.0, -1, -1, [node_features[node_id]]
	...            curr_src = node_id
	...        edge_type = 0
	...        if row[2] == 2018:
	...            edge_type = 1
	...        elif row[2] == 2019:
	...            edge_type = 2
	...        yield row[0], row[1], edge_type, float(row[3]), -1, -1, []

After preprocessing the dataset, we'll save it in a binary format to facilitate efficient graph operations. The data is stored in a subfolder named `deepgnn`.

.. code-block:: python

	>>> import json
	>>> import os
	>>> from deepgnn.graph_engine.snark.converter.writers import BinaryWriter
	>>> from deepgnn.graph_engine.snark.meta import BINARY_DATA_VERSION

	>>> binary_data = tempfile.mkdtemp()
	>>> writer = BinaryWriter(binary_data, suffix=0, watermark=-1)
	>>> writer.add(edge_iterator(edges, dataset.graph['node_feat']))
	>>> writer.close()

	>>> mjson = {
	...     "binary_data_version": BINARY_DATA_VERSION,
	...     "node_count": writer.node_count,
	...     "edge_count": writer.edge_count,
	...     "node_type_count": 1,
	...     "edge_type_count": 1,
	...     "node_feature_count": 1,
	...     "edge_feature_count": 0,
	...     "partitions": {
	...         "0": {
	...             "node_weight": [writer.node_count],
	...             "edge_weight": [writer.edge_count],
	...         }
	...     },
	...     "node_count_per_type": [writer.node_count],
	...     "edge_count_per_type": [writer.edge_count],
	...     "watermark": -1,
	... }
	>>> with open(osp.join(binary_data, "meta.json"), "w") as file:
	...     file.write(json.dumps(mjson))
	326

The final step involves querying specific node features to verify that it's ready for model training.

.. code-block:: python

	>>> from deepgnn.graph_engine.snark.local import Client
	>>> client = Client(binary_data, [0])
	>>> client.node_features(np.array([49077], dtype=np.int64), np.array([[0,4]], dtype=np.int32), np.float32)
	array([[-0.08541 ,  0.010725, -0.319365,  0.008517]], dtype=float32)
	>>> import shutil
	>>> shutil.rmtree(osp.join(temp_dir, "collab", "raw"))

Build Link Prediction Model
===========================

The primary objective is to design and implement a link prediction model. The model aims to estimate the probability of an edge (or link) existing between
any two nodes in a graph. For feature representation, we utilize node embeddings, generated through Graph Neural Networks (GNN), specifically, a GraphSAGE model.
The LinkPredictor class is a torch module that accepts embeddings from two nodes and predicts whether a link should exist between them. It consists of multiple
fully connected linear layers, ReLU activations, and dropout for regularization.


.. code-block:: python

	>>> import torch
	>>> import torch.nn.functional as F
	>>> from torch_geometric.nn import GraphSAGE
	>>> class LinkPredictor(torch.nn.Module):
	...    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
	...                 dropout):
	...        super(LinkPredictor, self).__init__()
	...
	...        self.lins = torch.nn.ModuleList([(torch.nn.Linear(in_channels, hidden_channels))])
	...        for _ in range(num_layers - 2):
	...            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
	...        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))
	...        self.dropout = dropout
	...
	...    def reset_parameters(self):
	...        for lin in self.lins:
	...            lin.reset_parameters()
	...
	...    def forward(self, x_i, x_j):
	...        x = x_i * x_j
	...        for lin in self.lins[:-1]:
	...            x = lin(x)
	...            x = F.relu(x)
	...            x = F.dropout(x, p=self.dropout, training=self.training)
	...        x = self.lins[-1](x)
	...        return torch.sigmoid(x)

We employ the GraphSAGE algorithm from the PyTorch Geometric(PyG) library to create node embeddings.

.. code-block:: python

	>>> config = {
	...     "feature_dim": 128,
	...     "hidden_channels": 256,
	...     "num_epochs": 2,
	...     "fanout": [5, 5],
	...     "batch_size": 64*1024,
	...     "num_nodes": writer.node_count,
	... }

	>>> model = GraphSAGE(
	...     config["feature_dim"],
	...     hidden_channels=config["hidden_channels"],
	...     num_layers=2,
	... )
	>>> predictor = LinkPredictor(config["hidden_channels"], config["hidden_channels"], 1, num_layers=3, dropout=0)
	>>> optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

Dataset iterator
================

We need to prepare an iterator over the dataset to provide minibatches of edges and features for training.
The SageDataset class serves as the iterator, sampling neighbors from the graph and creating an edge index for each edge in a minibatch.

.. code-block:: python

	>>> from torch.utils.data import IterableDataset, DataLoader
	>>> from deepgnn.graph_engine import Graph
	>>> class SageDataset(IterableDataset):
	...     def __init__(
	...         self,
	...         batch_size: int,
	...         fanout: list,
	...         graph: Graph,
	...         feature_dim: int,
	...         num_nodes: int,
	...         edge_list: np.array,
	...         generate_negs: bool = True,
	...     ):
	...         super(SageDataset, self).__init__()
	...         self.batch_size = batch_size
	...         self.num_nodes = num_nodes
	...         self.graph = graph
	...         self.fanout = fanout
	...         self.feature_dim = feature_dim
	...         self.edge_list = edge_list
	...         self.num_batches = -(-self.edge_list.shape[0] // batch_size)
	...         self.generate_negs = generate_negs
	...
	...     def __iter__(self):
	...         return map(self.query, range(self.num_batches))
	...
	...     def _make_edge_index(self, seed: np.array):
	...         fst_hop = self.graph.sample_neighbors(
	...             seed,
	...             np.array([0], dtype=np.int32),
	...             self.fanout[0],
	...         )
	...         fst_unique = np.unique(fst_hop[0].ravel())
	...         snd_hop = self.graph.sample_neighbors(
	...             fst_unique,
	...             np.array([0], dtype=np.int32),
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
	...         # to track labels from the seed array.
	...         unique_elements, first_occurrences, inverse_indices = np.unique(edges, return_index=True, return_inverse=True)
	...         reorder_by_first_occurrence = np.argsort(first_occurrences)
	...         unique_elements = unique_elements[reorder_by_first_occurrence]
	...         inverse_indices = np.argsort(reorder_by_first_occurrence)[inverse_indices]
	...
	...         edge_len = len(edges) // 2
	...         col = inverse_indices[:edge_len]
	...         row = inverse_indices[edge_len:]
	...         return unique_elements, col, row
	...
	...     def query(self, batch_index: int) -> tuple:
	...         start_idx = batch_index * self.batch_size
	...         end_idx = (batch_index + 1) * self.batch_size
	...         edges = self.edge_list[start_idx:end_idx, :]
	...         src = edges[:, 0]
	...         dst = edges[:, 1]
	...         num_pos = src.shape[0]
	...         num_neg = num_pos if self.generate_negs else 0
	...         neg_edges = np.random.randint(0, self.num_nodes - 1, size=2 * num_neg)
	...         seed = np.concatenate(
	...             [src, neg_edges[:num_neg], dst, neg_edges[num_neg:]], axis=0
	...         )
	...         edge_label = np.zeros(num_pos + num_neg)
	...         edge_label[:num_pos] = 1
	...         seed, inverse_seed = np.unique(seed, return_inverse=True)
	...         edge_label_index = inverse_seed.reshape((2, -1))
	...         nodes, cols, rows = self._make_edge_index(seed)
	...         feats = self.graph.node_features(
	...             nodes, np.array([[0, self.feature_dim]], dtype=np.int32), np.float32
	...         )
	...
	...         return (feats, cols, rows, edge_label_index, edge_label)

The function train orchestrates a single epoch of training. It iterates through the dataset,
makes predictions using both GraphSAGE and LinkPredictor, and computes the binary cross-entropy loss.

.. code-block:: python

	>>> def train(model, predictor, optimizer, dataset):
	...     model.train()
	...     total_loss = 0
	...     total_examples = 0
	...     train_dataloader = DataLoader(dataset)
	...     for batch in train_dataloader:
	...         node_features, cols, rows, edge_label_index, edge_label = (
	...             batch[0][0],
	...             batch[2][0],
	...             batch[1][0],
	...             batch[3][0],
	...             batch[4][0],
	...         )
	...         edge_index = torch.stack([cols, rows], dim=0)
	...         optimizer.zero_grad()
	...         h = model(node_features, edge_index)
	...         h_src = h[edge_label_index[0]]
	...         h_dst = h[edge_label_index[1]]
	...         pred = predictor(h_src, h_dst)
	...         loss = F.binary_cross_entropy_with_logits(pred.squeeze(), edge_label)
	...         loss.backward()
	...
	...         optimizer.step()
	...         num_examples = pred.size(0)
	...         total_examples += num_examples
	...         total_loss += float(loss) * num_examples
	...
	...     return total_loss / total_examples

Finally, we initiate the training process. The code iterates through multiple epochs, utilizing all the aforementioned
components, and prints out the loss for each epoch.

.. code-block:: python

	>>> for epoch in range(config["num_epochs"]):
	...     loss = train(
	...         model,
	...         predictor,
	...         optimizer,
	...         SageDataset(
	...             batch_size=config["batch_size"],
	...             fanout=config["fanout"],
	...             graph=client,
	...             feature_dim=config["feature_dim"],
	...             num_nodes=config["num_nodes"],
	...             edge_list=dataset.get_edge_split()['train']['edge'],
	...         ),
	...     )
	...     print(f"Epoch: {epoch:03d}, Loss: {loss:.4f}")
	Epoch: 000, Loss: 0...


Model Evaluation
================

Evaluating the performance of our link prediction model is the final step to understand its efficacy and reliability.
We'll be utilizing the OGB dataset for this purpose, specifically focusing on various edge splits to assess how well does
the model generalize to unseen data. We'll use the Evaluator class from OGB to compute the Hits@K score, a popular metric for link prediction tasks.
We calculate this score for multiple values of K, such as 10, 50, and 100. The Hits@K score indicates how often
the actual links appear within the top K ranked links predicted by the model.

.. code-block:: python

	>>> def create_eval_dataset(edges, config):
	...     return SageDataset(
	...         batch_size=config["batch_size"],
	...         fanout=config["fanout"],
	...         graph=client,
	...         feature_dim=config["feature_dim"],
	...         num_nodes=config["num_nodes"],
	...         edge_list=edges,
	...         generate_negs=False
	...     )

	>>> def evaluate_on_batch(model, predictor, batch):
	...    node_features, cols, rows, edge_label_index, _ = (batch[0][0], batch[2][0], batch[1][0], batch[3][0], batch[4][0])
	...    edge_index = torch.stack([cols, rows], dim=0)
	...    h = model(node_features, edge_index)
	...    return predictor(h[edge_label_index[0]], h[edge_label_index[1]]).squeeze().cpu()

	>>> def test(model, predictor, split_edge, evaluator, config):
	...    model.eval()
	...    predictor.eval()
	...
	...    edge_keys = ['train', 'valid', 'valid', 'test', 'test']
	...    edge_sub_keys = ['edge', 'edge', 'edge_neg', 'edge', 'edge_neg']
	...
	...    edge_preds = [[] for _ in range(5)]
	...
	...    for i, (key, sub_key) in enumerate(zip(edge_keys, edge_sub_keys)):
	...        edge_list = split_edge[key][sub_key]
	...        dataset = create_eval_dataset(edge_list, config)
	...
	...        for batch in DataLoader(dataset):
	...            edge_preds[i].append(evaluate_on_batch(model, predictor, batch))
	...
	...        edge_preds[i] = torch.cat(edge_preds[i], dim=0)
	...
	...    results = {}
	...    for K in [10, 50, 100]:
	...        evaluator.K = K
	...        results[f'Hits@{K}'] = (
	...            evaluator.eval({'y_pred_pos': edge_preds[0], 'y_pred_neg': edge_preds[2]})[f'hits@{K}'],
	...            evaluator.eval({'y_pred_pos': edge_preds[1], 'y_pred_neg': edge_preds[2]})[f'hits@{K}'],
	...            evaluator.eval({'y_pred_pos': edge_preds[3], 'y_pred_neg': edge_preds[4]})[f'hits@{K}']
	...        )
	...
	...    return results

We can now evaluate our model on the test splits with ogb's evaluator.

.. code-block:: python

	>>> from ogb.linkproppred import Evaluator
	>>> evaluator = Evaluator(name='ogbl-collab',)
	>>> test(model, predictor, dataset.get_edge_split(), evaluator, config)
	{'Hits@10'...
