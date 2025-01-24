# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Example of unsupervised graphsage model trained on Cora dataset."""

import os.path as osp
from typing import Any, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset
from sklearn.linear_model import LogisticRegression
from torch_geometric.nn import GraphSAGE

from deepgnn import get_logger
from deepgnn.graph_engine.data.citation import Cora
from deepgnn.graph_engine import Graph, SamplingStrategy


class SageDataset(IterableDataset):
    """Query to fetch graph data for the model."""

    def __init__(
        self,
        batch_size: int,
        fanout: list,
        graph: Graph,
        feature_dim: int,
        num_nodes: int,
    ):
        """Initialize graph query."""
        super(SageDataset, self).__init__()
        self.batch_size = 256
        self.num_nodes = num_nodes
        self.graph = graph
        self.fanout = fanout
        self.feature_dim = feature_dim

    def __iter__(self):
        """Return a subgraph from randomly sampled edges."""
        return map(self.query, range(0, self.graph.edge_count(0), self.batch_size))

    def _make_edge_index(self, seed: np.array):
        fst_hop = self.graph.sample_neighbors(
            seed,
            np.array([0], dtype=np.int32),
            self.fanout[0],
        )
        fst_unique = np.unique(fst_hop[0].ravel())
        snd_hop = self.graph.sample_neighbors(
            fst_unique,
            np.array([0], dtype=np.int32),
            self.fanout[1],
        )

        # Dedupe second hop edges for faster training.
        snd_edges = np.stack(
            [fst_unique.repeat(self.fanout[1]), snd_hop[0].ravel()], axis=1
        )
        snd_edges = np.unique(snd_edges, axis=0)
        edges = np.concatenate(
            [
                seed.repeat(self.fanout[0]),
                snd_edges[:, 0],
                fst_hop[0].ravel(),
                snd_edges[:, 1],
            ]
        )

        # np.unique returns sorted elements, but we need to preserve original order
        # to track labels from the seed array. We do it with argsort to get unique elements
        # in the original order and broadcasting to get inverse indices
        unique_nodes, first_occurrence_indices = np.unique(edges, return_index=True)
        sort_order = np.argsort(first_occurrence_indices)
        ordered_unique_nodes = unique_nodes[sort_order]
        broadcasted_comparison = edges[:, None] == ordered_unique_nodes
        inverse_indices = np.argmax(broadcasted_comparison, axis=1)

        edge_len = len(edges) // 2
        col = inverse_indices[:edge_len]
        row = inverse_indices[edge_len:]
        return ordered_unique_nodes, col, row

    def query(self, batch_id: int) -> tuple:
        """Fetch training data."""
        edges = self.graph.sample_edges(
            self.batch_size,
            np.array([0], dtype=np.int32),
            strategy=SamplingStrategy.Weighted,
        )
        src = edges[:, 0]
        dst = edges[:, 1]
        num_pos = src.shape[0]
        num_neg = num_pos
        neg_edges = np.random.randint(0, self.num_nodes - 1, size=2 * num_neg)
        seed = np.concatenate(
            [src, neg_edges[:num_neg], dst, neg_edges[num_neg:]], axis=0
        )
        edge_label = np.zeros(num_pos + num_neg)
        edge_label[:num_pos] = 1
        seed, inverse_seed = np.unique(seed, return_inverse=True)
        edge_label_index = inverse_seed.reshape((2, -1))
        nodes, cols, rows = self._make_edge_index(seed)
        feats = self.graph.node_features(
            nodes, np.array([[0, self.feature_dim]], dtype=np.int32), np.float32
        )

        # Return a tuple of tensors similar to NeighborSampler from PyG.
        subgraph: Tuple[Any, ...] = (feats, cols, rows, edge_label_index, edge_label)
        return subgraph


def train(model: GraphSAGE, optimizer: torch.optim.Optimizer, dataset: SageDataset):
    """Train the graphsage model for one epoch."""
    model.train()
    total_loss = 0
    train_dataloader = DataLoader(dataset)
    for batch in train_dataloader:
        node_features, cols, rows, edge_label_index, edge_label = (
            batch[0][0],
            batch[2][0],
            batch[1][0],
            batch[3][0],
            batch[4][0],
        )
        edge_index = torch.stack([cols, rows], dim=0)
        optimizer.zero_grad()
        h = model(node_features, edge_index)
        h_src = h[edge_label_index[0]]
        h_dst = h[edge_label_index[1]]
        pred = (h_src * h_dst).sum(dim=-1)
        loss = F.binary_cross_entropy_with_logits(pred, edge_label)
        loss.backward()
        optimizer.step()
        total_loss += float(loss) * pred.size(0)

    return total_loss


def _prepare_evaluation_data(
    config: dict,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    train_np = np.loadtxt(osp.join(config["data_dir"], "train.nodes"), dtype=np.int64)
    eval_batch = np.loadtxt(osp.join(config["data_dir"], "test.nodes"), dtype=np.int64)
    input_np = np.arange(config["num_nodes"], dtype=np.int64)
    input_nodes = torch.from_numpy(input_np).to(device)
    input_edges = graph.neighbors(input_np, np.array([0], dtype=np.int32))
    input_src = np.repeat(input_np, input_edges[3].astype(np.int64))
    input_dst = input_edges[0]
    input_features = graph.node_features(
        input_nodes, np.array([[0, config["feature_dim"]]], dtype=np.int32), np.float32
    )
    input_labels = graph.node_features(
        input_nodes, np.array([[1, 1]], dtype=np.int32), np.float32
    )
    edge_index_eval = torch.stack(
        [torch.from_numpy(input_src), torch.from_numpy(input_dst)], dim=0
    )
    train_mask = torch.zeros(len(input_nodes), dtype=torch.bool)
    train_mask[train_np] = True
    test_mask = torch.zeros(len(input_nodes), dtype=torch.bool)
    test_mask[eval_batch] = True
    return (
        torch.from_numpy(input_features),
        edge_index_eval,
        torch.from_numpy(input_labels),
        train_mask,
        test_mask,
    )


@torch.no_grad()
def test(
    model: GraphSAGE,
    features: torch.Tensor,
    edge_index_eval: torch.Tensor,
    labels: torch.Tensor,
    train_mask: torch.Tensor,
    test_mask: torch.Tensor,
) -> float:
    """Evaluate model performance on test dataset."""
    model.eval()
    out = model(features, edge_index_eval).cpu()
    clf = LogisticRegression()
    clf.fit(out[train_mask], labels[train_mask].squeeze())
    test_acc = clf.score(out[test_mask], labels[test_mask].squeeze())

    return test_acc


def _main(config: dict, graph: Graph):
    device = torch.device("cpu")
    model = GraphSAGE(
        config["feature_dim"],
        hidden_channels=64,
        num_layers=2,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # We'll use the same data for evaluation across epochs and can prepare it once.
    (
        node_features,
        edge_index_eval,
        node_labels,
        train_mask,
        test_mask,
    ) = _prepare_evaluation_data(config, device)

    test_acc = 0
    for epoch in range(config["num_epochs"]):
        loss = train(
            model,
            optimizer,
            SageDataset(
                batch_size=config["batch_size"],
                fanout=config["fanout"],
                graph=graph,
                feature_dim=config["feature_dim"],
                num_nodes=config["num_nodes"],
            ),
        )
        loss /= config["num_nodes"]
        test_acc = test(
            model, node_features, edge_index_eval, node_labels, train_mask, test_mask
        )
        get_logger().info(f"Epoch: {epoch:03d}, Loss: {loss:.4f}, Test: {test_acc:.4f}")
    assert test_acc > 0.6


if __name__ == "__main__":
    graph = CoraFull()
    _main(
        config={
            "batch_size": 256,
            "fanout": [5, 5],
            "feature_dim": 1433,
            "data_dir": graph.data_dir(),
            "num_nodes": graph.NUM_NODES,
            "num_epochs": 10,
        },
        graph=graph,
    )
