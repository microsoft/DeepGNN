# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import numpy as np

import ray
import ray.train as train
from ray.train.torch import TorchTrainer
from ray.air import session
from ray.air.config import ScalingConfig

import torch
from torch import nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn

from deepgnn.graph_engine import Graph
from deepgnn.graph_engine.snark.client import MemoryGraph
from deepgnn.graph_engine.data.citation import Cora
from deepgnn.pytorch.common import BaseMetric, MRR, F1Score


class PTGSupervisedGraphSage(nn.Module):
    """Supervised graphsage model implementation with torch geometric."""

    def __init__(
        self,
        num_classes: int,
        label_idx: int,
        label_dim: int,
        feature_dim: int,
        edge_type: int,
        fanouts: list,
        embed_dim: int = 128,
        metric: BaseMetric = MRR(),
    ):
        """Initialize a graphsage model for node classification."""
        super(PTGSupervisedGraphSage, self).__init__()
        self.num_classes = num_classes
        self.out_dim = num_classes
        self.metric = metric
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

        self.weight = nn.Parameter(
            torch.empty(embed_dim, num_classes, dtype=torch.float32)
        )
        self.metric = metric
        nn.init.xavier_uniform_(self.weight)

    def build_edges_tensor(self, N, K):
        """Build edge matrix."""
        nk = torch.arange((N * K).item(), dtype=torch.long, device=N.device)
        src = (nk // K).reshape(1, -1)
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
            n0_out, np.array([[self.feature_idx, self.feature_dim]]), self.feature_type
        )

        context["x0"] = x0.reshape((context["inputs"].shape[0], -1, self.feature_dim))
        context["out_1"] = np.array(
            [n1_out.shape[0]] * context["inputs"].shape[0]
        )  # Number of output nodes of layer 1
        context["out_2"] = np.array(
            [n2_out.shape[0]] * context["inputs"].shape[0]
        )  # Number of output nodes of layer 2
        return context

    def get_score(self, context: dict) -> torch.Tensor:  # type: ignore[override]
        """Generate scores for a list of nodes."""
        self.encode_feature(context)
        embeds = self.get_embedding(context)
        scores = torch.matmul(embeds, self.weight)
        return scores

    def metric_name(self):
        """Metric used for training."""
        return self.metric.name()

    def get_embedding(self, context: dict) -> torch.Tensor:  # type: ignore[override]
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

    train.torch.accelerate()
    train.torch.enable_reproducibility(seed=session.get_world_rank())

    model = PTGSupervisedGraphSage(
        num_classes=config["label_dim"],
        metric=F1Score(),
        label_idx=config["label_idx"],
        label_dim=config["label_dim"],
        feature_dim=config["feature_dim"],
        feature_idx=config["feature_idx"],
        feature_type=np.float32,
        edge_type=0,
        fanouts=[5, 5],
        feature_enc=None,
    )
    model = train.torch.prepare_model(model)
    model.train()

    optimizer = torch.optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config["learning_rate"] * session.get_world_size(),
    )
    optimizer = train.torch.prepare_optimizer(optimizer)

    g = MemoryGraph(config["data_dir"])
    max_id = g.node_count(0)
    dataset = ray.data.range(max_id).repartition(max_id // config["batch_size"])
    pipe = dataset.window(blocks_per_window=4).repeat(config["num_epochs"])
    dataset = pipe.map_batches(lambda idx: model.query(g, np.array(idx)))

    for i, epoch in enumerate(dataset.iter_epochs()):
        scores = []
        labels = []
        losses = []
        for batch in epoch.iter_torch_batches(batch_size=config["batch_size"]):
            loss, score, label = model(batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            scores.append(score)
            labels.append(label)
            losses.append(loss.item())

        if i % 10 == 0:
            print(
                f"Epoch {i:0>3d} {model.metric_name()}: {model.compute_metric(scores, labels).item():.4f} Loss: {np.mean(losses):.4f}"
            )

def _main():
    trainer = TorchTrainer(
        train_func,
        train_loop_config={
            # "data_dir": cora.data_dir(),
            "data_dir": "/tmp/citation/cora",
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
    assert result.metrics["MRR"] == 0.746


if __name__ == "__main__":
    _main()
