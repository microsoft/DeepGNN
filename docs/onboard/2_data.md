# Custom Caveman Dataset with GAT

In this guide we generate a [caveman graph with networkx](https://networkx.org/documentation/stable/reference/generated/networkx.generators.community.connected_caveman_graph.html?highlight=connected_caveman_graph#networkx.generators.community.connected_caveman_graph), convert it to a dataset that the graph engine can read and train a [Graph Attention Network(GAT)](https://arxiv.org/abs/1710.10903) to predict what cluster a node belongs to.

## Download Notebook

Download the coresponding jupyter notebook, <a href="~/tutorials/pytorch/2_data.ipynb" download="2_data.ipynb" target="_blank">caveman_data.ipynb</a>. Right click to save as `2_data.ipynb`.

## Custom Dataset

In order to create a caveman graph dataset that we can train on, we will first generate a caveman graph with networkx, then save it in our json format and finally convert those files to a binary format that the snark graph engine can read.

### Generate

First we use networkx to generate a caveman graph.
The caveman graph consists of M clusters each with N nodes, every cluster has a single edge randomly rewired to another group.

```python
>>> import networkx as nx
>>> import random
>>> random.seed(0)
>>> num_clusters = 5
>>> num_nodes_in_cluster = 10
>>> g = nx.connected_caveman_graph(num_clusters, num_nodes_in_cluster)

```

### Save

Then we save the graph in [json format](~/advanced/data_spec.md) which includes files `graph.json` and `meta.json`. `graph.json` contains all feature and connectivity information. `meta.json` describes the high level information of the graph: the number of node and edge types and the number of typed attributes for nodes and edges.

Full json format, [here](~/advanced/data_spec.md).

```python
>>> import os
>>> working_dir = "/tmp/caveman"
>>> try:
...     os.mkdir(working_dir)
... except FileExistsError:
...     pass

>>> import json
>>> from pathlib import Path
>>> test_pct = .2  # Percent of nodes in test set
>>> nodes = []
>>> data = ""
>>> for node_id in g:
...     cluster_id = float(node_id // num_nodes_in_cluster)
...     normalized_cluster = cluster_id / num_clusters - 0.4
...     node = {
...         "node_id": node_id,  # int
...         "node_type": 1 if random.random() < test_pct else 0,  # int
...         "node_weight": 1.,  # float
...         "uint64_feature": None,  # None or {"feature id": ["int", "..."], "...": "..."},
...         "float_feature": {
...             "0": [
...                 0.02 * random.uniform(0, 1) + 2.5 * normalized_cluster - 0.01,
...                 random.uniform(0, 1),
...             ],
...             "1": [cluster_id],
...         },  #  {"feature id": ["float", "..."], "...": "..."},
...         "binary_feature": None,  # {"feature id": "string", "...": "..."},
...         "edge": [
...             {
...                 "src_id": node_id,  # int
...                 "dst_id": neighbor_id,  # int
...                 "edge_type": 0,  # int
...                 "weight": 1.0,  # float
...                 #     "uint64_feature": {"feature id": ["int", "..."], "...": ["int", "..."]},
...                 #     "float_feature": {"feature id": ["float", "..."], "...": ["float", "..."]},
...                 #     "binary_feature": {"feature id": "string", "...": "..."}
...             }
...             for neighbor_id in nx.neighbors(g, node_id)
...         ],
...         "neighbor": {
...             "0": dict(
...                 [
...                     (str(neighbor_id), 1.0)
...                     for neighbor_id in nx.neighbors(g, node_id)
...                 ]
...             )
...         },  # {"edge type": {"neighbor id": "weight", "...": "..."}, "...": "..."}
...     }
...     data += json.dumps(node) + "\n"
...     nodes.append(node)

>>> data_filename = f"{working_dir}/graph.json"
>>> with open(data_filename, "w+") as f:
...     f.write(data)
4...

>>> meta = '{"node_type_num": 2, \
...             "node_float_feature_num": 2, \
...             "node_binary_feature_num": 0, \
...             "node_uint64_feature_num": 0, \
...             "edge_type_num": 1, \
...             "edge_float_feature_num": 0, \
...             "edge_binary_feature_num": 0, \
...             "edge_uint64_feature_num": 0}'

>>> meta_filename = f"{working_dir}/meta.json"
>>> with open(meta_filename, "w+") as f:
...     f.write(meta)
302

```

### Convert

Finally, in order to use this dataset we convert the files from json to binary with snark's `convert` function. `convert` can also split the nodes among multiple files, called partitions. Each graph engine will load all partitions, with the more partitions creating less variance between node lookups with a small per-partition overhead.

```python
>>> import deepgnn.graph_engine.snark.convert as convert
>>> from deepgnn.graph_engine.snark.decoders import LinearDecoder
>>> partitions = 1
>>> convert.MultiWorkersConverter(
...     graph_path=data_filename,
...     meta_path=meta_filename,
...     partition_count=partitions,
...     output_dir=working_dir,
...     decoder_type=LinearDecoder,
... ).convert()

```

or via command line,

```bash
python -m deepgnn.graph_engine.snark.convert -d /tmp/caveman/graph.json -m /tmp/caveman/meta.json -o /tmp/caveman -p 1
```

At this point the dataset is complete and ready to be used in a model.

## Using dataset in GAT

Using this Graph Attention Network, we can accurately guess which cluster a specific node belongs to based on its features and the features of nodes it is connected to.
More about GAT in the [GAT Node Classification Example](~/onboard/1_node_class.md).

We copy the GAT model from [DeepGNN's examples directory](https://github.com/microsoft/DeepGNN/blob/main/examples/pytorch/gat). Though the dataset iterator and a few parameters have been changed to accomodate the dataset, the code block after this one contains the code that is different.

```python
>>> from typing import List
>>> from dataclasses import dataclass
>>> import argparse
>>> import numpy as np
>>> import torch
>>> import torch.nn as nn
>>> import torch.nn.functional as F

>>> from deepgnn.pytorch.common import Accuracy
>>> from deepgnn.pytorch.modeling.base_model import BaseModel
>>> from deepgnn.pytorch.nn.gat_conv import GATConv

>>> from deepgnn.graph_engine import Graph, FeatureType, graph_ops

>>> from deepgnn import str2list_int
>>> from deepgnn.pytorch.common.utils import set_seed
>>> from deepgnn.pytorch.common.dataset import TorchDeepGNNDataset
>>> from deepgnn.pytorch.modeling import BaseModel
>>> from deepgnn.pytorch.training import run_dist
>>> from deepgnn.graph_engine import GENodeSampler, FileNodeSampler, GraphEngineBackend  # Note

>>> @dataclass
... class GATQueryParameter:
...     neighbor_edge_types: np.array
...     feature_idx: int
...     feature_dim: int
...     label_idx: int
...     label_dim: int
...     feature_type: FeatureType = FeatureType.FLOAT
...     label_type: FeatureType = FeatureType.FLOAT
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
...         super().__init__(FeatureType.FLOAT, 0, 0, None)
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
...         # fmt: off
...         nodes, feat, mask, labels, edges, edges_value, adj_shape = inputs
...         nodes = torch.squeeze(nodes)                # [N], N: num of nodes in subgraph
...         feat = torch.squeeze(feat)                  # [N, F]
...         mask = torch.squeeze(mask)                  # [N]
...         labels = torch.squeeze(labels)              # [N]
...         edges = torch.squeeze(edges)                # [X, 2], X: num of edges in subgraph
...         edges_value = torch.squeeze(edges_value)    # [X]
...         adj_shape = torch.squeeze(adj_shape)        # [2]
...         # fmt: on
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

>>> def init_args(parser):
...     parser.add_argument("--head_num", type=str2list_int, default="8,1", help="the number of attention headers.")
...     parser.add_argument("--hidden_dim", type=int, default=8, help="hidden layer dimension.")
...     parser.add_argument("--num_classes", type=int, default=-1, help="number of classes for category")
...     parser.add_argument("--ffd_drop", type=float, default=0.0, help="feature dropout rate.")
...     parser.add_argument("--attn_drop", type=float, default=0.0, help="attention layer dropout rate.")
...     parser.add_argument("--l2_coef", type=float, default=0.0005, help="l2 loss")
...     parser.add_argument("--neighbor_edge_types", type=str2list_int, default="0", help="Graph Edge for attention encoder.",)
...     parser.add_argument("--eval_file", default="", type=str, help="")

```

The below code has been modified from the original, changed lines are marked with a comment containing the original value.

```python
>>> def create_dataset(
...     args: argparse.Namespace,
...     model: BaseModel,
...     rank: int = 0,
...     world_size: int = 1,
...     backend: GraphEngineBackend = None,
... ):
...     return TorchDeepGNNDataset(
...         sampler_class=GENodeSampler,  # FileNodeSampler,
...         node_types=np.array([0]),  # None
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
...     "--num_classes", str(num_clusters),
...     "--batch_size", "10",
...     "--learning_rate", ".005",
...     "--num_epochs", "10",
...     "--log_by_steps", "6",
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

Finally we train the model to predict which cluster a given node belongs to via run_dist. We expect the loss to decrease with epochs, the number of epochs and learning rate can be adjusted to better achieve this.

```python
>>> run_dist(
...     init_model_fn=create_model,
...     init_dataset_fn=create_dataset,
...     init_optimizer_fn=create_optimizer,
...     init_args_fn=init_args,
... )

```

Now with the python file set, we can train a model and save the parameters for future testing and inference. There are a few expected parameters that need to be filled out for the graph engine, model and trainer. For training we use node_type 0 and evaluation uses node_type 1 so that we have a separate training and testing set. Notably `MODEL_DIR` is where model parameters will be saved and is defined as a bash variable so it can be reused after training.

In scenarios that use a custom dataset there are a few parameters to pay attention to including the feature and label's index and dimension as well as the number of classes. They are defined as follows,

* feature_idx: int, Index of the vector within graph.json's `float_feature` that should be used as the training feature.
* feature_dim: int, Size of the feature vector.
* label_idx: int, Index of the vector within graph.json's `float_feature` that should be used as the expected label.
* label_dim: int, Size of the label vector.
* num_classes: int, Number of possible labels, in the caveman dataset this is equal to the number of clusters.

```bash
MODEL_DIR=$HOME/tmp/gat_$(date +"%Y%m%d_%H%M%N")
python docs/testfile2.py --data_dir /tmp/caveman --mode train --node_type 0 --trainer base \
    --backend snark --graph_type local --converter skip \
    --feature_idx 0 --feature_dim 2 --label_idx 1 --label_dim 1 --num_classes 5 \
    --batch_size 10 --learning_rate .005 --num_epochs 10 \
    --log_by_steps 6 --use_per_step_metrics \
    --model_dir $MODEL_DIR --metric_dir $MODEL_DIR --save_path $MODEL_DIR

[2021-10-31 19:59:45,947] {utils.py:111} INFO - parameter count: 673
[2021-10-31 19:59:45,963] {logging_utils.py:76} INFO - Training worker started. Model: GAT.
[2021-10-31 19:59:46,912] {samplers.py:148} INFO - Node Count ([0]): 50
[2021-10-31 19:59:46,915] {samplers.py:148} INFO - Node Count ([0]): 50
[2021-10-31 19:59:47,089] {trainer.py:230} INFO - [1,0] epoch: 0; step: 00006; loss: 1.4296; Accuracy: 0.5556; time: 1.7619s
...
[2021-10-31 19:59:49,196] {trainer.py:230} INFO - [1,0] epoch: 8; step: 00006; loss: 0.8166; Accuracy: 1.0000; time: 0.2467s
[2021-10-31 19:59:49,210] {trainer.py:439} INFO - [1,0] Saved checkpoint to /home/user/tmp/gat_20211031_1959/gnnmodel-009-000000.pt.
[2021-10-31 19:59:49,235] {samplers.py:148} INFO - Node Count ([0]): 50
[2021-10-31 19:59:49,245] {samplers.py:148} INFO - Node Count ([0]): 50
[2021-10-31 19:59:49,452] {trainer.py:230} INFO - [1,0] epoch: 9; step: 00006; loss: 0.7262; Accuracy: 1.0000; time: 0.2562s
[2021-10-31 19:59:49,466] {trainer.py:439} INFO - [1,0] Saved checkpoint to /home/user/tmp/gat_20211031_1959/gnnmodel-010-000000.pt.
[2021-10-31 19:59:49,467] {logging_utils.py:76} INFO - Training worker finished. Model: GAT.
```

## Evaluate

Finally that we have a trained model and have its parameters saved to `MODEL_DIR`, we can check its test accuracy by switching the run mode to evaluate, node_type to 1 and learning_rate to 0.

```bash
python docs/testfile2.py --data_dir /tmp/caveman --mode evaluate --node_type 1 --trainer base \
    --backend snark --graph_type local --converter skip \
    --feature_idx 0 --feature_dim 2 --label_idx 1 --label_dim 1 --num_classes 5 \
    --batch_size 140 --learning_rate .0 --num_epochs 10 \
    --log_by_steps 6 --use_per_step_metrics \
    --model_dir $MODEL_DIR --metric_dir $MODEL_DIR --save_path $MODEL_DIR

[2021-10-31 20:07:11,442] {logging_utils.py:76} INFO - Training worker started. Model: GAT.
[2021-10-31 20:07:11,475] {samplers.py:148} INFO - Node Count ([0]): 50
[2021-10-31 20:07:11,486] {samplers.py:148} INFO - Node Count ([0]): 50
[2021-10-31 20:07:11,534] {trainer.py:312} INFO - [1,0] Evaluation Accuracy: 0.7849; data size: 93;
[2021-10-31 20:07:11,536] {trainer_hvd.py:24} INFO - [1,0] AllReduced Accuracy: 0.7849; loss: 0.7085
[2021-10-31 20:07:11,536] {logging_utils.py:76} INFO - Training worker finished. Model: GAT.
```


#### [Prev Page](~/onboard/1_node_class.md) <div style="display: inline;float: right">[Next Page](~/onboard/3_linkpred.md)</div>
