# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Example of GCN model trained on Cora dataset."""

from typing import Tuple
from dataclasses import dataclass

import numpy as np
import ray
import ray.train as train
from ray.train.torch import TorchTrainer
from ray.air import session
from ray.air.config import ScalingConfig
from sklearn.metrics import f1_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCN2Conv
from torch.utils.data import DataLoader, IterableDataset
from torch_sparse.tensor import SparseTensor

from deepgnn.graph_engine import Graph, graph_ops
from deepgnn.graph_engine.snark.distributed import Server, Client as DistributedClient
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
    batch_size: int = 20
    feature_type: np.dtype = np.float32
    label_type: np.dtype = np.float32
    num_hops: int = 2


class GCNDataset(IterableDataset):
    """Dataset to fetch graph data for the model.

    Implements IterableDataset such that inputs and outputs are numpy arays.
    Torch converts them to tensors in DataLoader wrapping this dataset.

    Args:
    p (GCNQueryParameter): query configuration
    inputs (np.array): input nodes, shape [batch_size, num_epochs*batches_per_epoch]
    graph (Graph): graph object
    """

    def __init__(self, query: GCNQueryParameter, inputs: np.array, graph: Graph):
        """Initialize graph query."""
        super(GCNDataset, self).__init__()
        self.query = query
        self.graph = graph
        self.inputs = inputs
        self.label_meta = np.array([[query.label_idx, query.label_dim]], np.int32)
        self.feat_meta = np.array([[query.feature_idx, query.feature_dim]], np.int32)

    def __iter__(self):
        """
        Create iterator over inputs.

        We shuffle the inputs at the beginning of each epoch and drop last nodes not fitting
        into a batch.
        """
        np.random.shuffle(self.inputs)
        worker_info = torch.utils.data.get_worker_info()
        batches = self.inputs
        last_elements = len(self.inputs) % self.query.batch_size
        if last_elements > 0:
            batches = batches[:-last_elements]
        batches = batches.reshape(-1, self.query.batch_size)

        if worker_info is None:
            return map(self.__call__, batches)

        # split workload for distributed training
        return map(self.__call__, batches[worker_info.id :: worker_info.num_workers])

    def __call__(self, inputs: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Fetch graph data."""
        nodes, edges, _ = graph_ops.sub_graph(
            self.graph,
            inputs,
            edge_types=self.query.neighbor_edge_types,
            num_hops=self.query.num_hops,
            self_loop=True,
            undirected=True,
            return_edges=True,
        )
        feat = self.graph.node_features(nodes, self.feat_meta, self.query.feature_type)
        y = self.graph.node_features(nodes, self.label_meta, self.query.label_type)
        # One hot encoding for integers in Cora graph. Not needed for Reddit/PPI graphs and can return y directly.
        labels = np.zeros(
            (nodes.size, self.query.num_classes), dtype=self.query.label_type
        )
        labels[np.arange(nodes.size), y[:, 0].astype(np.int32)] = 1

        edges = np.transpose(edges)

        return (feat, edges, labels)


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
            h = F.dropout(x, self.dropout, training=self.training)
            h = conv(h, x_0, adj)
            x = h + x
            x = x.relu()

        x = F.dropout(x, self.dropout, training=self.training)
        return self.lins[1](x)


def train_func(config: dict):
    """Training loop for ray trainer."""
    train.torch.enable_reproducibility(seed=session.get_world_rank())
    train_query = GCNQueryParameter(
        neighbor_edge_types=np.array([0], np.int32),
        **config["graph_query"],
    )
    test_query = GCNQueryParameter(
        neighbor_edge_types=np.array(
            [0, 1], np.int32
        ),  # edges to nodes in test datasets usually have type 1.
        **config["graph_query"],
    )
    model = GCN(
        hidden_channels=2048,
        num_layers=9,
        alpha=0.5,
        theta=1.0,
        feature_dim=train_query.feature_dim,
        num_classes=train_query.num_classes,
        shared_weights=False,
        dropout=0.2,
    ).to(config["device"])

    model = train.torch.prepare_model(model)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=0.001 * session.get_world_size(),
    )

    g = config["get_graph"]()
    train_np = np.loadtxt(f"{config['data_dir']}/train.nodes", dtype=np.int64)
    eval_batch = np.loadtxt(f"{config['data_dir']}/test.nodes", dtype=np.int64)

    train_dataloader = DataLoader(GCNDataset(train_query, train_np, g))
    train_dataloader = train.torch.prepare_data_loader(train_dataloader)
    eval_dataloader = DataLoader(GCNDataset(test_query, eval_batch, g))
    eval_dataloader = train.torch.prepare_data_loader(eval_dataloader)

    for _ in range(config["num_epochs"]):
        total_loss = 0
        num_examples = 0
        model.train()
        for batch in train_dataloader:
            sparse = SparseTensor.from_edge_index(batch[1][0])
            loss = config["loss_func"](model(batch[0][0], sparse), batch[2][0])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            num_examples += len(batch[2][0])
            total_loss += loss.item() * len(batch[2][0])

        model.eval()
        ys, preds = [], []
        for batch in eval_dataloader:
            ys.append(batch[2][0])
            adj = SparseTensor.from_edge_index(batch[1][0])
            out = model(batch[0][0].to(config["device"]), adj.to(config["device"]))
            preds.append((out > 0).float().cpu())

        y, pred = torch.cat(ys, dim=0).numpy(), torch.cat(preds, dim=0).numpy()
        f1 = f1_score(y, pred, average="micro") if pred.sum() > 0 else 0
        session.report(
            {
                "f1_score": f1.item(),
                "loss": total_loss / num_examples,
            },
        )


if __name__ == "__main__":
    ray.init(num_cpus=3)

    address = "localhost:9999"
    dataset = Cora()
    s = Server(address, dataset.data_dir(), 0, 1)

    def get_graph():
        """Create a new client for each worker."""
        return DistributedClient([address])

    trainer = TorchTrainer(
        train_func,
        train_loop_config={
            "get_graph": get_graph,
            "data_dir": dataset.data_dir(),
            "device": torch.device("cpu"),
            "num_epochs": 1,
            "loss_func": nn.BCEWithLogitsLoss(),
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
    assert result.metrics["f1_score"] > 0.3
