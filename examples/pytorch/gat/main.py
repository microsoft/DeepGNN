# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from typing import List, Any
from dataclasses import dataclass

import numpy as np


from deepgnn.pytorch.common.metrics import Accuracy
from deepgnn.graph_engine import Graph, graph_ops
from deepgnn.graph_engine.snark.distributed import Server, Client as DistributedClient
from deepgnn.graph_engine.data.citation import Cora
# print("Starting example")
# cora = Cora(output_dir="/tmp/cora")

import ray
import ray.train as train
from ray.train.torch import TorchTrainer
from ray.air import session
from ray.air.config import ScalingConfig

import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
import os



@dataclass
class GATQueryParameter:
    """Configuration for graph query."""

    neighbor_edge_types: np.ndarray
    feature_idx: int
    feature_dim: int
    label_idx: int
    label_dim: int
    feature_type: np.dtype = np.float32
    label_type: np.dtype = np.float32
    num_hops: int = 2


class GATQuery:
    """Query to fetch graph data for the model."""

    def __init__(self, p: GATQueryParameter):
        """Initialize graph query."""
        self.p = p
        self.label_meta = np.array([[p.label_idx, p.label_dim]], np.int32)
        self.feat_meta = np.array([[p.feature_idx, p.feature_dim]], np.int32)

    def __call__(self, graph: Graph, inputs: np.ndarray) -> tuple:
        """Fetch training data."""
        nodes, edges, src_idx = graph_ops.sub_graph(
            graph,
            inputs,
            edge_types=self.p.neighbor_edge_types,
            num_hops=self.p.num_hops,
            self_loop=True,
            undirected=True,
            return_edges=True,
        )
        input_mask = np.zeros(nodes.size, np.bool_)
        input_mask[src_idx] = True
        feat = graph.node_features(nodes, self.feat_meta, self.p.feature_type)
        label = graph.node_features(nodes, self.label_meta, self.p.label_type)
        label = label.astype(np.int32)
        edges = np.transpose(edges)

        graph_tensor: List[Any] = [nodes, feat, edges, input_mask, label]
        return graph_tensor


class GAT(nn.Module):
    """GAT model."""

    def __init__(
        self,
        in_dim: int,
        q_param: GATQueryParameter,
        head_num: List = [8, 1],
        hidden_dim: int = 8,
        num_classes: int = -1,
    ):
        """Initialize model."""
        super(GAT, self).__init__()
        self.query = GATQuery(q_param)
        self.num_classes = num_classes
        self.out_dim = num_classes
        self.metric = Accuracy()
        self.xent = nn.CrossEntropyLoss()
        self.conv1 = GATConv(
            in_channels=in_dim,
            out_channels=hidden_dim,
            heads=head_num[0],
            dropout=0.6,
            concat=True,
        )
        layer0_output_dim = head_num[0] * hidden_dim
        self.conv2 = GATConv(
            in_channels=layer0_output_dim,
            out_channels=self.out_dim,
            heads=1,
            dropout=0.6,
            concat=False,
        )

    def forward(self, inputs):
        """Calculate loss, make predictions and fetch labels."""
        nodes, feat, edge_index, mask, label = inputs
        nodes = torch.squeeze(nodes.to(torch.int32))  # [N]
        feat = torch.squeeze(feat.to(torch.float32))  # [N, F]
        edge_index = torch.squeeze(edge_index.to(torch.int32))  # [2, X]
        mask = torch.squeeze(mask.to(torch.bool))  # [N]
        labels = torch.squeeze(label.to(torch.int32))  # [N]

        x = feat
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        scores = self.conv2(x, edge_index)
        labels = labels.type(torch.int64)
        labels = labels[mask]  # [batch_size]
        scores = scores[mask]  # [batch_size]
        pred = scores.argmax(dim=1)
        loss = self.xent(scores, labels)

        return loss, pred, labels


def train_func(config: dict):
    """Training loop for ray trainer."""

    train.torch.enable_reproducibility(seed=session.get_world_rank())
    p = GATQueryParameter(
        neighbor_edge_types=np.array([0], np.int32),
        feature_idx=config["feature_idx"],
        feature_dim=config["feature_dim"],
        label_idx=config["label_idx"],
        label_dim=config["label_dim"],
    )
    model = GAT(
        in_dim=config["feature_dim"],
        head_num=[8, 1],
        hidden_dim=8,
        num_classes=config["num_classes"],
        q_param=p,
    )
    model = train.torch.prepare_model(model)

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=0.005 * session.get_world_size(),
        weight_decay=0.0005,
    )

    g = config["get_graph"]()
    batch_size = 140
    train_dataset = ray.data.from_numpy(
        np.loadtxt(f"{config['data_dir']}/train.nodes", dtype=np.int64)
    )

    train_pipe = train_dataset.repeat(config["num_epochs"])
    eval_batch = np.loadtxt(f"{config['data_dir']}/test.nodes", dtype=np.int64)

    train_batches = [
        batch for batch in train_pipe.iter_torch_batches(batch_size=batch_size)
    ]

    for batch in train_batches:
        model.train()
        train_tensor = model.query(g, batch.detach().numpy())
        loss, score, label = model([torch.from_numpy(a) for a in train_tensor])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        model.eval()

        eval_tensor = model.query(g, eval_batch)
        loss, score, label = model([torch.from_numpy(a) for a in eval_tensor])

        session.report(
            {
                model.metric.name(): model.metric.compute(score, label).item(),
                "loss": loss.detach().numpy(),
            },
        )

import os

def _main():
    os.environ["RAY_PROFILING"] = "1"
    ray.init(num_cpus=3) #, log_to_driver=False)

    address = "localhost:9999"
    # s = Server(address, cora.data_dir(), 0, 1)
    s = Server(address, "/tmp/cora", 0, 1)

    def get_graph():
        return DistributedClient([address])

    trainer = TorchTrainer(
        train_func,
        train_loop_config={
            "get_graph": get_graph,
            # "data_dir": cora.data_dir(),
            "data_dir": "/tmp/cora",
            "num_epochs": 200,
            "feature_idx": 0,
            "feature_dim": 1433,
            "label_idx": 1,
            "label_dim": 1,
            "num_classes": 7,
        },
        scaling_config=ScalingConfig(num_workers=1),
    )
    result = trainer.fit()
    ray.timeline(filename="/tmp/timeline.json")
    assert result.metrics["Accuracy"] == 0.746


if __name__ == "__main__":
    _main()
