# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Example of graphsage model trained on Cora dataset."""
from typing import Tuple

import numpy as np
from sklearn.metrics import f1_score
import torch
from torch import nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn

from deepgnn.graph_engine import Graph, QueryOutput
from deepgnn.graph_engine.snark.local import Client as MemoryGraph
from deepgnn.graph_engine.data.citation import Cora


class PTGSupervisedGraphSage(nn.Module):
    """Supervised graphsage model implementation with torch geometric."""

    def __init__(
        self,
        num_classes: int,
        label_idx: int,
        label_dim: int,
        feature_idx: int,
        feature_dim: int,
        edge_type: int,
        fanouts: list,
        embed_dim: int = 128,
    ):
        """Initialize a graphsage model for node classification."""
        super(PTGSupervisedGraphSage, self).__init__()
        self.num_classes = num_classes
        self.out_dim = num_classes
        self.xent = nn.CrossEntropyLoss()

        # only 2 hops are allowed.
        assert len(fanouts) == 2
        self.fanouts = fanouts
        self.edge_type = edge_type

        conv_model = pyg_nn.SAGEConv
        self.convs = nn.ModuleList()
        self.convs.append(conv_model(feature_dim, embed_dim))
        self.convs.append(conv_model(embed_dim, embed_dim))
        self.label_idx = label_idx
        self.label_dim = label_dim
        self.feature_idx = feature_idx
        self.feature_dim = feature_dim

        self.weight = nn.Parameter(
            torch.empty(embed_dim, num_classes, dtype=torch.float32)
        )
        nn.init.xavier_uniform_(self.weight)

    def build_edges_tensor(self, N, K):
        """Build edge matrix."""
        nk = torch.arange((N * K).item(), dtype=torch.long, device=N.device)
        src = torch.div(nk, K, rounding_mode="floor").reshape(1, -1)
        dst = (N + nk).reshape(1, -1)
        elist = torch.cat([src, dst], dim=0)
        return elist

    def query(self, graph: Graph, inputs: np.ndarray) -> dict:
        """Query graph for training data."""
        context = {"inputs": inputs}
        context["label"] = graph.node_features(
            context["inputs"],
            np.array([[self.label_idx, self.label_dim]]),
            np.float32,
        )

        n2_out = context[
            "inputs"
        ].flatten()  # Output nodes of 2nd (final) layer of convolution
        # input nodes of 2nd layer of convolution (besides the output nodes themselves)
        n2_in = graph.sample_neighbors(n2_out, self.edge_type, self.fanouts[1])[
            0
        ].flatten()
        #  output nodes of first layer of convolution (all nodes that affect output of 2nd layer)
        n1_out = np.concatenate([n2_out, n2_in])
        # input nodes to 1st layer of convolution (besides the output)
        n1_in = graph.sample_neighbors(n1_out, self.edge_type, self.fanouts[0])[
            0
        ].flatten()
        # Nodes for which we need features (layer 0)
        n0_out = np.concatenate([n1_out, n1_in])
        x0 = graph.node_features(
            n0_out, np.array([[self.feature_idx, self.feature_dim]]), np.float32
        )

        context["x0"] = x0.reshape((context["inputs"].shape[0], -1, self.feature_dim))
        context["out_1"] = np.array(
            [n1_out.shape[0]] * context["inputs"].shape[0]
        )  # Number of output nodes of layer 1
        context["out_2"] = np.array(
            [n2_out.shape[0]] * context["inputs"].shape[0]
        )  # Number of output nodes of layer 2
        return context

    def _loss_inner(
        self, context: QueryOutput
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Cross entropy loss for a list of nodes."""
        if isinstance(context, dict):
            labels = context["label"].squeeze()
        elif isinstance(context, torch.Tensor):
            labels = context.squeeze()  # type: ignore
        else:
            raise TypeError("Invalid input type.")
        scores = self.get_score(context)
        labels = labels.to(torch.int64)
        return (
            self.xent(
                scores,
                labels,
            ),
            scores.argmax(dim=1),
            labels,
        )

    def forward(
        self, context: QueryOutput
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return cross entropy loss."""
        return self._loss_inner(context)

    # type: ignore[override]
    def get_score(self, context: dict) -> torch.Tensor:
        """Generate scores for a list of nodes."""
        embeds = self.get_embedding(context)
        scores = torch.matmul(embeds, self.weight)
        return scores

    def metric_name(self):
        """Metric used for training."""
        return self.metric.name()

    # type: ignore[override]
    def get_embedding(self, context: dict) -> torch.Tensor:
        """Generate embedding."""
        out_1 = context["out_1"][0]
        out_2 = context["out_2"][0]
        try:
            out_1 = out_1[0]
            out_2 = out_2[0]
        except IndexError:
            pass
        edges_1 = self.build_edges_tensor(out_1, self.fanouts[0])  # Edges for 1st layer
        x1 = self.convs[0](
            context["x0"].reshape((-1, context["x0"].shape[-1])), edges_1
        )[
            :out_1, :
        ]  # Output of 1st layer (cut out 2 hop nodes)
        x1 = F.relu(x1)
        edges_2 = self.build_edges_tensor(out_2, self.fanouts[1])  # Edges for 2nd layer
        x2 = self.convs[1](x1, edges_2)[
            :out_2, :
        ]  # Output of second layer (nodes for which loss is computed)
        x2 = F.relu(x2)
        return x2


def train_func(config: dict):
    """Training loop for ray trainer."""
    model = PTGSupervisedGraphSage(
        num_classes=config["num_classes"],
        label_idx=config["label_idx"],
        label_dim=config["label_dim"],
        feature_dim=config["feature_dim"],
        feature_idx=config["feature_idx"],
        edge_type=0,
        fanouts=[10, 10],
    )
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    g = MemoryGraph(config["data_dir"], partitions=[0])
    train_dataset = np.loadtxt(f"{config['data_dir']}/train.nodes", dtype=np.int64)
    eval_batch = np.loadtxt(f"{config['data_dir']}/test.nodes", dtype=np.int64)

    for epoch in range(config["num_epochs"]):
        losses = []
        np.random.shuffle(train_dataset)
        for batch in np.split(
            train_dataset, train_dataset.size // config["batch_size"]
        ):
            train_tuple = model.query(g, batch)
            loss, score, label = model(
                {k: torch.from_numpy(train_tuple[k]) for k in train_tuple}
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses = loss.item()

        if epoch % 10 == 0:
            model.eval()
            eval_tuple = model.query(g, eval_batch)
            loss, score, label = model(
                {k: torch.from_numpy(eval_tuple[k]) for k in eval_tuple}
            )
            f1 = f1_score(
                label.squeeze(), score.detach().cpu().numpy(), average="micro"
            ).item()
            print(f"Epoch {epoch:0>3d} F1Score: {f1:.4f} Loss: {losses:.4f}")
            model.train()


def _main():
    cora = Cora()
    result = train_func(
        {
            "data_dir": cora.data_dir(),
            "num_epochs": 500,
            "feature_idx": 0,
            "feature_dim": 1433,
            "label_idx": 1,
            "label_dim": 1,
            "num_classes": 7,
            "batch_size": 140,
        },
    )
    assert result.metrics["F1Score"] == 0.746


if __name__ == "__main__":
    _main()
