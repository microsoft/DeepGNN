# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Example of GCN model trained on Cora dataset."""

from typing import Tuple
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCN2Conv
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_sparse.tensor import SparseTensor

from deepgnn.graph_engine import Graph, graph_ops, get_logger
from deepgnn.graph_engine.data.citation import Cora


@dataclass
class GCNQueryParameter:
    """Configuration for graph query to fetch data for model."""

    neighbor_edge_types: np.ndarray
    feature_idx: int
    feature_dim: int
    label_idx: int
    label_dim: int
    num_classes: int
    num_hops: int = 2


def query(
    inputs: np.ndarray, graph: Graph, param: GCNQueryParameter
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fetch graph data."""
    nodes, edges, _ = graph_ops.sub_graph(
        graph,
        inputs,
        edge_types=param.neighbor_edge_types,
        num_hops=param.num_hops,
        self_loop=True,
        undirected=True,
        return_edges=True,
    )

    # Extract features with a single call to graph engine.
    feat_labels = graph.node_features(
        nodes,
        np.array(
            [
                [param.feature_idx, param.feature_dim],
                [param.label_idx, param.label_dim],
            ],
            dtype=np.int32,
        ),
        np.float32,
    )

    feat_labels = torch.from_numpy(feat_labels)
    adj = SparseTensor.from_edge_index(
        torch.from_numpy(np.transpose(edges)),
    )

    return (
        feat_labels[:, : param.feature_dim],
        gcn_norm(adj),
        feat_labels[:, param.feature_dim : param.feature_dim + param.label_dim]
        .to(torch.int64)
        .squeeze(),
    )


class GCN(nn.Module):
    """
    Example GCN model from PyG.

    Original example:
    https://github.com/pyg-team/pytorch_geometric/blob/02cc18d9d6841ecda4eb61eebb48b86f1aaae477/examples/gcn2_ppi.py#L23
    """

    def __init__(
        self,
        hidden_channels: int,
        num_layers: int,
        alpha: float,
        theta: float,
        feature_dim: int,
        num_classes: int,
        shared_weights: bool = True,
        dropout: float = 0.0,
    ):
        """Initialize model."""
        super().__init__()

        self.lins = nn.ModuleList()
        self.lins.append(nn.Linear(feature_dim, hidden_channels))
        self.lins.append(nn.Linear(hidden_channels, num_classes))
        self.dropout = dropout
        self.convs = nn.ModuleList()
        for layer in range(num_layers):
            self.convs.append(
                GCN2Conv(
                    hidden_channels,
                    alpha,
                    theta,
                    layer + 1,
                    shared_weights,
                    normalize=False,
                )
            )

    def forward(self, x: torch.Tensor, adj: SparseTensor):
        """Forward pass."""
        x = F.dropout(x, self.dropout, training=self.training)
        x = x_0 = self.lins[0](x).relu()

        for conv in self.convs:
            x = F.dropout(x, self.dropout, training=self.training)
            x = conv(x, x_0, adj)
            x = x.relu()

        x = F.dropout(x, self.dropout, training=self.training)
        x = self.lins[1](x)

        return x.log_softmax(dim=-1)


def _train_func(config: dict):
    query_param = GCNQueryParameter(
        neighbor_edge_types=np.array(
            [0], np.int32
        ),  # edges to nodes in test datasets usually have type 1, but we can use 0 for Cora.
        **config["graph_query"],
    )
    model = GCN(
        feature_dim=query_param.feature_dim,
        num_classes=query_param.num_classes,
        hidden_channels=64,
        num_layers=64,
        alpha=0.1,
        theta=0.5,
        shared_weights=True,
        dropout=0.6,
    ).to(config["device"])

    optimizer = torch.optim.Adam(
        [
            dict(params=model.convs.parameters(), weight_decay=0.01),
            dict(params=model.lins.parameters(), weight_decay=5e-4),
        ],
        lr=0.01,
    )

    g = config["graph"]
    masks = {}
    for mask in ["train", "test"]:
        full_set = torch.zeros(g.NUM_NODES, dtype=torch.bool)
        nodes = torch.from_numpy(
            np.loadtxt(f"{config['data_dir']}/{mask}.nodes", dtype=np.int64)
        )
        full_set[nodes] = True
        masks[mask] = full_set

    # GCN doesn't need any sampling, so we can just query the full graph.
    feat, adj, labels = query(np.arange(g.NUM_NODES, dtype=np.int64), g, query_param)
    accuracy: float = 0

    for epoch in range(config["num_epochs"]):
        total_loss = 0
        num_examples = 0
        model.train()
        optimizer.zero_grad()
        out = model(feat, adj)
        loss = F.nll_loss(out[masks["train"]], labels[masks["train"]])
        loss.backward()
        optimizer.step()
        num_examples += len(labels[masks["train"]])
        total_loss += loss.item() * len(labels[masks["train"]])
        if epoch % 10 == 0:
            model.eval()
            ys = labels[masks["test"]]
            preds = out[masks["test"]].argmax(dim=-1).cpu()
            accuracy = int((ys == preds).sum()) / int(len(ys))
            get_logger().info(
                f"Epoch: {epoch:>2d}; Loss: {(total_loss/num_examples):>4.2f}; Accuracy: {accuracy:.4f};"
            )

    assert accuracy > 0.45


if __name__ == "__main__":
    torch.random.manual_seed(42)
    np.random.seed(42)
    dataset = Cora()
    _train_func(
        {
            "graph": dataset,
            "data_dir": dataset.data_dir(),
            "device": torch.device("cpu"),
            "num_epochs": 101,
            "graph_query": {
                "feature_idx": 0,
                "feature_dim": dataset.FEATURE_DIM,
                "label_idx": 1,
                "label_dim": 1,
                "num_classes": dataset.NUM_CLASSES,
            },
        }
    )
