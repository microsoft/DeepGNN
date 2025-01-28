# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Example of GAT model trained on Cora dataset."""

from typing import List, Any, Tuple
from dataclasses import dataclass
import os.path as osp

import numpy as np
import ray
import ray.train as train
from ray.train.torch import TorchTrainer
from ray.air import session
from ray.air.config import ScalingConfig
from sklearn.metrics import accuracy_score
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset
from torch_geometric.nn import GATConv

from deepgnn.graph_engine.data.citation import Cora
from deepgnn.graph_engine import Graph, graph_ops
from deepgnn.graph_engine.snark.distributed import Server, Client as DistributedClient


@dataclass
class GATQueryParameter:
    """Configuration for graph query."""

    neighbor_edge_types: np.ndarray
    feature_idx: int
    feature_dim: int
    label_idx: int
    label_dim: int
    num_classes: int
    batch_size: int = 70
    feature_type: np.dtype = np.float32
    label_type: np.dtype = np.float32
    num_hops: int = 2


class GATDataset(IterableDataset):
    """Query to fetch graph data for the model."""

    def __init__(self, query_param: GATQueryParameter, inputs: np.array, graph: Graph):
        """Initialize graph query."""
        super(GATDataset, self).__init__()
        self.query_param = query_param
        self.graph = graph
        self.inputs = inputs
        self.label_meta = np.array(
            [[query_param.label_idx, query_param.label_dim]], np.int32
        )
        self.feat_meta = np.array(
            [[query_param.feature_idx, query_param.feature_dim]], np.int32
        )

    def __iter__(self):
        """
        Create iterator over inputs.

        We shuffle the inputs at the beginning of each epoch and drop last nodes not fitting
        into a batch.
        """
        np.random.shuffle(self.inputs)
        worker_info = torch.utils.data.get_worker_info()
        batches = self.inputs
        last_elements = len(self.inputs) % self.query_param.batch_size
        if last_elements > 0:
            batches = batches[:-last_elements]
        batches = batches.reshape(-1, self.query_param.batch_size)

        if worker_info is None:
            return map(self.query, batches)

        # split workload for distributed training
        return map(self.query, batches[worker_info.id :: worker_info.num_workers])

    def query(self, inputs: np.ndarray) -> tuple:
        """Fetch training data."""
        nodes, edges, src_idx = graph_ops.sub_graph(
            self.graph,
            inputs,
            edge_types=self.query_param.neighbor_edge_types,
            num_hops=self.query_param.num_hops,
            self_loop=True,
            undirected=True,
            return_edges=True,
        )
        input_mask = np.zeros(nodes.size, np.bool_)
        input_mask[src_idx] = True
        feat = self.graph.node_features(
            nodes, self.feat_meta, self.query_param.feature_type
        )
        label = self.graph.node_features(
            nodes, self.label_meta, self.query_param.label_type
        )
        label = label.astype(np.int32)
        edges = np.transpose(edges)

        graph_tensor: Tuple[Any, ...] = (nodes, feat, edges, input_mask, label)
        return graph_tensor


class GAT(nn.Module):
    """GAT model."""

    def __init__(
        self,
        in_dim: int,
        head_num: List = [8, 1],
        hidden_dim: int = 8,
        num_classes: int = -1,
    ):
        """Initialize model."""
        super(GAT, self).__init__()
        self.num_classes = num_classes
        self.out_dim = num_classes
        self.xent = nn.CrossEntropyLoss()
        self.conv1 = GATConv(
            in_channels=in_dim,
            out_channels=hidden_dim,
            heads=head_num[0],
            dropout=0.6,
        )
        layer0_output_dim = head_num[0] * hidden_dim
        self.conv2 = GATConv(
            in_channels=layer0_output_dim,
            out_channels=self.out_dim,
            heads=1,
            dropout=0.6,
            concat=False,
        )

    def forward(self, inputs: Tuple[Any, ...]):
        """Calculate loss, make predictions and fetch labels."""
        nodes, feat, edge_index, mask, label = inputs
        nodes = torch.squeeze(nodes.to(torch.int32))  # [N]
        feat = torch.squeeze(feat.to(torch.float32))  # [N, F]
        edge_index = torch.squeeze(edge_index.to(torch.int32))  # [2, X]
        mask = torch.squeeze(mask.to(torch.bool))  # [N]
        labels = torch.squeeze(label.to(torch.int64))  # [N]

        x = feat
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        scores = self.conv2(x, edge_index)
        labels = labels[mask]  # [batch_size]
        scores = scores[mask]  # [batch_size]
        pred = scores.argmax(dim=1)
        loss = self.xent(scores, labels)

        return loss, pred, labels


def _train_func(config: dict):
    train.torch.enable_reproducibility(seed=session.get_world_rank())
    train_query = GATQueryParameter(
        neighbor_edge_types=np.array([0], np.int32),
        **config["graph_query"],
    )
    test_query = GATQueryParameter(
        neighbor_edge_types=np.array(
            [0, 1], np.int32
        ),  # edges to nodes in test datasets usually have type 1.
        **config["graph_query"],
    )
    model = GAT(
        in_dim=train_query.feature_dim,
        head_num=[8, 1],
        hidden_dim=8,
        num_classes=train_query.num_classes,
    ).to(config["device"])
    model = train.torch.prepare_model(model)

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=0.005 * session.get_world_size(),
        weight_decay=0.0005,
    )

    g = config["get_graph"]()
    train_np = np.loadtxt(osp.join(config["data_dir"], "train.nodes"), dtype=np.int64)
    eval_batch = np.loadtxt(osp.join(config["data_dir"], "test.nodes"), dtype=np.int64)

    train_dataloader = DataLoader(GATDataset(train_query, train_np, g))
    train_dataloader = train.torch.prepare_data_loader(train_dataloader)
    eval_dataloader = DataLoader(GATDataset(test_query, eval_batch, g))
    eval_dataloader = train.torch.prepare_data_loader(eval_dataloader)

    model.train()
    for _ in range(config["num_epochs"]):
        for batch in train_dataloader:
            loss, _, _ = model(batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.eval()
    eval_tensor = next(iter(eval_dataloader))
    _, score, label = model(eval_tensor)
    accuracy = torch.tensor(
        accuracy_score(y_true=label.cpu(), y_pred=score.detach().cpu().numpy())
    )
    session.report({"accuracy": accuracy.item()})


if __name__ == "__main__":
    ray.init(num_cpus=3)

    address = "localhost:9999"
    dataset = Cora()
    s = Server(address, dataset.data_dir(), 0, 1)

    def get_graph():
        """Create a new client for each worker."""
        return DistributedClient([address])

    trainer = TorchTrainer(
        _train_func,
        train_loop_config={
            "get_graph": get_graph,
            "data_dir": dataset.data_dir(),
            "device": torch.device("cpu"),
            "num_epochs": 200,
            "graph_query": {
                "feature_idx": 0,
                "feature_dim": dataset.FEATURE_DIM,
                "label_idx": 1,
                "label_dim": 1,
                "num_classes": dataset.NUM_CLASSES,
            },
        },
        scaling_config=ScalingConfig(num_workers=2),
    )

    result = trainer.fit()
    assert result.metrics["accuracy"] > 0.7
