# Custom Link Prediction Model

In this guide we build a Link Prediction model to predict whether there is a link between any two nodes of the [caveman dataset](https://networkx.org/documentation/stable/reference/generated/networkx.generators.community.connected_caveman_graph.html?highlight=connected_caveman_graph#networkx.generators.community.connected_caveman_graph).

If you are interested in the high level DeepGNN training flow, see the overiew guide, [here](~/onboard/1_node_class.md). This guide focuses on building a custom model.

## Download Notebook

Download the coresponding jupyter notebook, <a href="~/tutorials/pytorch/3_link_pred.ipynb" download="3_link_pred.ipynb" target="_blank">link_pred.ipynb</a>. Right click to save as `3_link_pred.ipynb`.

## Generate Dataset

A caveman graph consists of M clusters each with N nodes, each cluster has a single edge randomly rewired to another group.
We generate this dataset with networkx, save it to json format and use snark to convert it to a binary format that the graph engine can read. See the full caveman dataset implementation details, [here](~/onboard/2_data.md).

```python
>>> import networkx as nx
>>> import random
>>> random.seed(0)

>>> num_clusters = 30
>>> num_nodes_in_cluster = 12
>>> g = nx.connected_caveman_graph(num_clusters, num_nodes_in_cluster)

>>> test_ratio = 0.2 # Ratio of edges in a test dataset
>>> edge_list = list(nx.edges(g))
>>> random.shuffle(edge_list)
>>> test_cutoff = int(test_ratio * len(edge_list))
>>> test_dataset = set(edge_list[:test_cutoff])
>>> train_dataset = set(edge_list[test_cutoff:])

>>> import os
>>> working_dir = "/tmp/caveman"
>>> try:
...     os.mkdir(working_dir)
... except FileExistsError:
...     pass

>>> import json
>>> nodes = []
>>> data = ""
>>> for node_id in g:
...     cluster_id = float(node_id // num_nodes_in_cluster)
...     train_list = []
...     test_list = []
...     for neighbor_id in nx.neighbors(g, node_id):
...         if (node_id, neighbor_id) in train_dataset:
...             train_list.append(neighbor_id)
...         else:
...             test_list.append(neighbor_id)
...     node = {
...         "node_weight": 1,
...         "node_id": node_id,
...         "node_type": 0,
...         "uint64_feature": None,
...         "float_feature": {
...             "0": [float(cluster_id)/num_clusters],
...         },
...         "binary_feature": None,
...         "edge": [
...             {
...                 "src_id": node_id,
...                 "dst_id": neighbor_id,
...                 "edge_type": 0,
...                 "weight": 1.0,
...             }
...             for neighbor_id in train_list
...         ] + [
...             {
...                 "src_id": node_id,
...                 "dst_id": neighbor_id,
...                 "edge_type": 1,
...                 "weight": 1.0,
...             }
...             for neighbor_id in test_list
...         ],
...         "neighbor": {
...             "0": dict([(str(neighbor_id), 1.0) for neighbor_id in train_list]),
...             "1": dict([(str(neighbor_id), 1.0) for neighbor_id in test_list])
...         },
...     }
...     data += json.dumps(node) + "\n"
...     nodes.append(node)

>>> data_filename = working_dir + "/data.json"
>>> with open(data_filename, "w+") as f:
...     f.write(data)
357456

>>> meta = '{"node_float_feature_num": 1, \
...          "edge_binary_feature_num": 0, \
...          "edge_type_num": 2, \
...          "edge_float_feature_num": 0, \
...          "node_type_num": 2, \
...          "node_uint64_feature_num": 0, \
...          "node_binary_feature_num": 0, \
...          "edge_uint64_feature_num": 0}'
>>> meta_filename = working_dir + "/meta.json"
>>> with open(meta_filename, "w+") as f:
...     f.write(meta)
281

>>> import deepgnn.graph_engine.snark.convert as convert
>>> from deepgnn.graph_engine.snark.decoders import JsonDecoder
>>> partitions = 1
>>> convert.MultiWorkersConverter(
...     graph_path=data_filename,
...     meta_path=meta_filename,
...     partition_count=partitions,
...     output_dir=working_dir,
...     decoder_class=JsonDecoder(),
... ).convert()

```

## Build Link Prediction Model

Our goal is to create a model capable of predicting whether an edge exists between any two nodes based on their own and their neighbor's feature vectors.

```python
>>> from dataclasses import dataclass
>>> import argparse
>>> import numpy as np
>>> import torch
>>> from deepgnn.pytorch.modeling.base_model import BaseModel
>>> from deepgnn.graph_engine import FeatureType, SamplingStrategy, GEEdgeSampler, GraphEngineBackend
>>> from deepgnn.pytorch.common.utils import set_seed
>>> from deepgnn.pytorch.common.dataset import TorchDeepGNNDataset
>>> from deepgnn.pytorch.modeling import BaseModel
>>> from deepgnn.pytorch.training import run_dist
>>> from deepgnn.pytorch.common.metrics import F1Score

```

Query is the interface between the model and graph database. It uses the graph engine API to perform graph functions like `node_features` and `sample_neighbors`, for a full reference on this interface see, [this guide](~/advanced/graph_engine.md). Typically Query is initialized by the model as `self.q` so its functions may also be used ad-hoc by the model.

In this example, the query function will generate a set of positive and negative samples that represent real and fake links respectively. Positive samples are real edges taken directly from the sampler while negative samples have the same source nodes as those sampled combined with random destination nodes. For both sets of samples, query will take their set of source and destination nodes and indivudally grab their features, then fetch and aggregate their neighbor's features, therefore rendering four outputs for each set of samples: source node features, destination node features, aggregated source node neighbor features and aggregated destination node neighbor features. This return value contains all graph information needed by the forward function for a single batch.

```python
>>> @dataclass
... class LinkPredictionQueryParameter:
...     neighbor_edge_types: np.array
...     feature_idx: int
...     feature_dim: int
...     label_idx: int
...     label_dim: int
...     feature_type: FeatureType = FeatureType.FLOAT
...     label_type: FeatureType = FeatureType.FLOAT


>>> class LinkPredictionQuery:
...     def __init__(self, p: LinkPredictionQueryParameter):
...         self.p = p
...         self.label_meta = np.array([[p.label_idx, p.label_dim]], np.int32)
...         self.feat_meta = np.array([[p.feature_idx, p.feature_dim]], np.int32)
...
...     def _query(self, g, nodes, edge_types):
...         # Sample neighbors for every input node
...         try:
...             nodes = nodes.detach().numpy()
...         except Exception:
...             pass
...         nbs = g.sample_neighbors(
...             nodes=nodes.astype(dtype=np.int64),
...             edge_types=edge_types)[0]
...
...         # Extract features for all neighbors
...         nbs_features = g.node_features(
...             nodes=nbs.reshape(-1),
...             features=self.feat_meta,
...             feature_type=self.p.feature_type)
...
...         # reshape the feature tensor to [nodes, neighbors, features]
...         # and aggregate along neighbors dimension.
...         nbs_agg = nbs_features.reshape(list(nbs.shape)+[self.p.feature_dim]).mean(1)
...         node_features = g.node_features(
...             nodes=nodes.astype(dtype=np.int64),
...             features=self.feat_meta,
...             feature_type=self.p.feature_type,
...         )
...         return node_features, nbs_agg
...
...     def query_training(self, ge, edges, edge_types = np.array([0], dtype=np.int32)):
...         edges = torch.Tensor(edges[:, :2]).long()
...         src, src_nbs = self._query(ge, edges[:, 0], edge_types)
...         dst, dst_nbs = self._query(ge, edges[:, 1], edge_types)
...         context = [edges, src, src_nbs, dst, dst_nbs]
...
...         # Prepare negative examples: edges between source nodes and random nodes
...         dim = len(edges)
...         source_nodes = torch.as_tensor(edges[:, 0], dtype=torch.int64).reshape(1, dim)
...         random_nodes = ge.sample_nodes(dim, node_types=0, strategy=SamplingStrategy.Weighted).reshape(1, dim)
...         neg_inputs = torch.cat((source_nodes, torch.tensor(random_nodes)), axis=1)
...         src, src_nbs = self._query(ge, neg_inputs[:, 0], edge_types)
...         dst, dst_nbs = self._query(ge, neg_inputs[:, 1], edge_types)
...         context += [edges, src, src_nbs, dst, dst_nbs]
...
...         return context

```

The model init and forward look the same as any other pytorch model, though instead of inhereting `torch.nn.Module`, we base off of `deepgnn.pytorch.modeling.base_model.BaseModel` which itself is a torch module with DeepGNN's specific interface. The forward function is expected to return three values: the batch loss, the model predictions for the given nodes and the expected labels for the given nodes.

In this example,
* `get_score` estimates the likelihood of a link existing between the nodes given. It accomplishes this by taking the difference between source and destination node features and aggregating these results. The final output is maped to `[0, 1]` interval with a sigmoid function. This function is used by `forward` as a helper function.
* `forward` scores the connection likelihood for the positive and negative samples given and computes the loss as the sum of binary cross entropies of each sample set. The intuition behind this algorithm is the feature difference for nodes in the same cluster should be `0` while nodes from different clusters should be strictly larger than `0`.
* `metric` is specified in init and is used to determine the accuracy of the model based on the model predictions and expected labels returned by `forward`. Here we use the F1Score to evaluate the model, which is the simple binary accuracy.

```python
>>> class LinkPrediction(BaseModel):
...     def __init__(self, q_param):
...         self.q = LinkPredictionQuery(q_param)
...         super().__init__(
...             feature_type=q_param.feature_type,
...             feature_idx=q_param.feature_idx,
...             feature_dim=q_param.feature_dim,
...             feature_enc=None
...         )
...         self.feat_dim = q_param.feature_dim
...         self.embed_dim = 16
...         self.encode = torch.nn.Parameter(torch.FloatTensor(self.embed_dim, 2*self.feat_dim))
...         self.weight = torch.nn.Parameter(torch.FloatTensor(1, self.embed_dim))
...         torch.nn.init.xavier_uniform_(self.weight)
...         torch.nn.init.xavier_uniform_(self.encode)
...
...         self.metric = F1Score()
...
...     def get_score(self, context: torch.Tensor, edge_types: np.array):
...         edges, src, src_nbs, dst, dst_nbs = context
...         src, src_nbs, dst, dst_nbs = [v.detach().numpy() for v in (src, src_nbs, dst, dst_nbs)]
...
...         diff, diff_nbs = np.fabs(dst-src), np.fabs(dst_nbs-src_nbs)
...         final = np.concatenate((diff, diff_nbs), axis=1)
...
...         embed = self.encode.mm(torch.tensor(final).t())
...         score = self.weight.mm(embed)
...         return torch.sigmoid(score)
...
...     def forward(self, context: torch.Tensor, edge_types: np.array = np.array([0], dtype=np.int32)):
...         context = [v.squeeze(0) for v in context]
...         pos_label = self.get_score(context[:5], edge_types)
...         true_xent = torch.nn.functional.binary_cross_entropy(
...                 target=torch.ones_like(pos_label), input=pos_label, reduction="mean"
...             )
...
...         neg_label = self.get_score(context[5:], edge_types)
...         negative_xent = torch.nn.functional.binary_cross_entropy(
...             target=torch.zeros_like(neg_label), input=neg_label, reduction="mean"
...         )
...
...         loss = torch.sum(true_xent) + torch.sum(negative_xent)
...
...         pred = (torch.cat((pos_label.reshape((-1)), neg_label.reshape((-1)))) >= .5)
...         label = torch.cat((torch.ones_like(pos_label, dtype=bool).reshape((-1)), torch.zeros_like(neg_label, dtype=bool).reshape((-1))))
...         return loss, pred, label

```

Now we define the `create_` functions for use with `run_dist`. These functions allow command line arguments to be used in object creation. Each has a simple interface and requires little code changes per different model. The optimizers world_size parameter is the number of workers.

```python
>>> def create_model(args: argparse.Namespace):
...     if args.seed:
...         set_seed(args.seed)
...
...     p = LinkPredictionQueryParameter(
...             neighbor_edge_types=np.array([0], np.int32),
...             feature_idx=0,
...             feature_dim=2,
...             label_idx=1,
...             label_dim=1,
...         )
...
...     return LinkPrediction(p)

>>> def create_optimizer(args: argparse.Namespace, model: BaseModel, world_size: int):
...     return torch.optim.Adam(
...         filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001
...     )

```

Here we define `create_dataset` which allows command line argument parameterization of the dataset iterator.
The rank parameter is the index of the worker, world_size is the total number of workers and the backend is chosen via command line arguments `backend` and `graph_type`. Notably we use the `GEEdgeSampler` which uses the backend to sample edges with types `edge_types`, otherwise in our [node classification example](~/onboard/1_node_class.md) we use `FileNodeSampler` which loads `sample_files` and generates samples from them.

```python
>>> def create_dataset(
...     args: argparse.Namespace,
...     model: BaseModel,
...     rank: int = 0,
...     world_size: int = 1,
...     backend: GraphEngineBackend = None,
... ):
...     return TorchDeepGNNDataset(
...         sampler_class=GEEdgeSampler,
...         edge_types=np.array([0]),
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

```

### Arguments

`init_args` registers any model specific arguments with `parser` as the argparse parser. In this example we do not need any extra arguments but the commented out code can be used for reference to add integer and list of integer arguments respectively

```python
>>> def init_args(parser):
...     #parser.add_argument("--hidden_dim", type=int, default=8, help="hidden layer dimension.")
...     #parser.add_argument("--head_num", type=str2list_int, default="8,1", help="the number of attention headers.")
...     pass

```

NOTE Below code block is for jupyter notebooks only.

```python
>>> try:
...     init_args_base
... except NameError:
...     init_args_base = init_args

>>> MODEL_DIR = f"~/tmp/gat_{np.random.randint(9999999)}"
>>> arg_list = [
...     "--data_dir", "/tmp/caveman",
...     "--mode", "train",
...     "--trainer", "base",
...     "--backend", "snark",
...     "--graph_type", "local",
...     "--converter", "skip",
...     "--node_type", "0",
...     "--feature_idx", "0",
...     "--feature_dim", "2",
...     "--label_idx", "1",
...     "--label_dim", "1",
...     "--batch_size", "64",
...     "--learning_rate", ".001",
...     "--num_epochs", "100",
...     "--log_by_steps", "16",
...     "--data_parallel_num", "0",
...     "--use_per_step_metrics",
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

```

## Train

Finally we train the model to predict whether an edge exists between any two nodes to via run_dist. We expect the loss to decrease with epochs, the number of epochs and learning rate can be adjusted to better achieve this.

```python
>>> run_dist(
...     init_model_fn=create_model,
...     init_dataset_fn=create_dataset,
...     init_optimizer_fn=create_optimizer,
...     init_args_fn=init_args,
... )

```

Now with the python file set, we can train a model and save the parameters for future testing and inference. There are a few expected parameters that need to be filled out for the graph engine, model and trainer. Notably `MODEL_DIR` is where model parameters will be saved and is defined as a bash variable so it can be reused after training.

```bash
MODEL_DIR=$HOME/tmp/link_pred_$(date +"%Y%m%d_%H%M%N")
python docs/testfile3.py --data_dir /tmp/caveman --mode train --trainer base \
    --backend snark --graph_type local --converter skip \
    --feature_idx 0 --feature_dim 2 --label_idx 1 --label_dim 1 \
    --batch_size 64 --learning_rate .001 --num_epochs 100 --log_by_steps 16 \
    --model_dir $MODEL_DIR --metric_dir $MODEL_DIR --save_path $MODEL_DIR

[2021-11-01 00:48:52,770] {trainer.py:230} INFO - [1,0] epoch: 0; step: 00016; loss: 1.3883; time: 0.4259s
[2021-11-01 00:48:52,881] {trainer.py:439} INFO - [1,0] Saved checkpoint to /home/user/tmp/link_pred_20211101_0048/gnnmodel-001-000000.pt.
[2021-11-01 00:48:52,905] {samplers.py:204} INFO - Edge Count ([0]): 1584
[2021-11-01 00:48:52,921] {samplers.py:204} INFO - Edge Count ([0]): 1584
[2021-11-01 00:48:53,124] {trainer.py:230} INFO - [1,0] epoch: 1; step: 00016; loss: 1.4114; time: 0.3535s
[2021-11-01 00:48:53,243] {trainer.py:439} INFO - [1,0] Saved checkpoint to /home/user/tmp/link_pred_20211101_0048/gnnmodel-002-000000.pt.
...
[2021-11-01 00:49:24,439] {trainer.py:230} INFO - [1,0] epoch: 98; step: 00016; loss: 1.0057; time: 0.3048s
[2021-11-01 00:49:24,549] {trainer.py:439} INFO - [1,0] Saved checkpoint to /home/user/tmp/link_pred_20211101_0048/gnnmodel-099-000000.pt.
[2021-11-01 00:49:24,574] {samplers.py:204} INFO - Edge Count ([0]): 1584
[2021-11-01 00:49:24,583] {samplers.py:204} INFO - Edge Count ([0]): 1584
[2021-11-01 00:49:24,751] {trainer.py:230} INFO - [1,0] epoch: 99; step: 00016; loss: 1.2776; time: 0.3121s
```

## Evaluate

Finally that we have a trained model and have its parameters saved to `MODEL_DIR`, we can check its test accuracy by switching the run mode to evaluate and learning_rate to 0.

```bash
python docs/testfile3.py --data_dir /tmp/caveman --mode evaluate --trainer base \
    --backend snark --graph_type local --converter skip \
    --feature_idx 0 --feature_dim 2 --label_idx 1 --label_dim 1 \
    --batch_size 256 --learning_rate .0 --num_epochs 10 --log_by_steps 16 \
    --model_dir $MODEL_DIR --metric_dir $MODEL_DIR --save_path $MODEL_DIR

[2021-11-01 00:49:29,374] {logging_utils.py:76} INFO - Training worker started. Model: LinkPrediction.
[2021-11-01 00:49:29,403] {samplers.py:204} INFO - Edge Count ([0]): 1584
[2021-11-01 00:49:29,415] {samplers.py:204} INFO - Edge Count ([0]): 1584
[2021-11-01 00:49:29,532] {trainer.py:312} INFO - [1,0] Evaluation F1Score: 0.7870; data size: 2056;
[2021-11-01 00:49:29,534] {trainer_hvd.py:24} INFO - [1,0] AllReduced F1Score: 0.7870; loss: 1.0242
[2021-11-01 00:49:29,535] {logging_utils.py:76} INFO - Training worker finished. Model: LinkPrediction.
```

#### [Prev Page](~/onboard/2_data.md)
