# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""This example demonstrates how to use ray to submit a job to Azure ML using ray-on-aml."""
from typing import List, Any
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn.functional as F

import ray
import ray.train as train
from ray.train.torch import TorchTrainer
from ray.air import session
from ray.air.config import ScalingConfig, RunConfig

from torch_geometric.nn import GATConv
from deepgnn.graph_engine import Graph, graph_ops
from deepgnn.pytorch.modeling import BaseModel
from deepgnn.graph_engine.data.citation import Cora
from deepgnn.graph_engine.snark.distributed import Server, Client as DistributedClient
from deepgnn.graph_engine.utils import serialize, deserialize
from deepgnn.pytorch.common import Accuracy


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

    def query_training(self, graph: Graph, inputs: np.ndarray) -> tuple:
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
        return serialize(graph_tensor, inputs.size)


class GAT(BaseModel):
    """GAT model."""

    def __init__(
        self,
        in_dim: int,
        q_param: GATQueryParameter,
        head_num: List = [8, 1],
        hidden_dim: int = 8,
        num_classes: int = -1,
        ffd_drop: float = 0.0,
        attn_drop: float = 0.0,
    ):
        """Initialize model."""
        self.q = GATQuery(q_param)
        super().__init__(np.float32, 0, 0, None)
        self.num_classes = num_classes

        self.out_dim = num_classes

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

        self.metric = Accuracy()

    def forward(self, inputs):
        """Calculate loss, make predictions and fetch labels."""
        nodes, feat, edge_index, mask, label = deserialize(inputs)
        # fmt: off
        nodes = torch.squeeze(nodes.to(torch.int32))                # [N]
        feat = torch.squeeze(feat.to(torch.float32))                  # [N, F]
        edge_index = torch.squeeze(edge_index.to(torch.int32))      # [2, X]
        mask = torch.squeeze(mask.to(torch.bool))                  # [N]
        labels = torch.squeeze(label.to(torch.int32))               # [N]
        # fmt: on

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
    cora = Cora()

    address = "localhost:9999"
    _ = Server(address, cora.data_dir(), 0, 1)

    def get_graph():
        return DistributedClient([address])

    config["get_graph"] = get_graph

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
        ffd_drop=0.6,
        attn_drop=0.6,
        q_param=p,
    )
    model = train.torch.prepare_model(model)

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=0.005 * session.get_world_size(),
        weight_decay=0.0005,
    )
    optimizer = train.torch.prepare_optimizer(optimizer)

    g = config["get_graph"]()
    batch_size = 140
    train_dataset = ray.data.read_text(f"{cora.data_dir()}/train.nodes")
    train_dataset = train_dataset.repartition(train_dataset.count() // batch_size)
    train_pipe = train_dataset.window(blocks_per_window=4).repeat(config["num_epochs"])
    train_pipe = train_pipe.map_batches(
        lambda idx: model.q.query_training(g, np.array(idx))
    )

    test_dataset = ray.data.read_text(f"{cora.data_dir()}/test.nodes")
    test_dataset = test_dataset.repartition(1)
    test_dataset = test_dataset.map_batches(
        lambda idx: model.q.query_training(g, np.array(idx))
    )
    test_dataset_iter = test_dataset.repeat(config["num_epochs"]).iter_epochs()

    for epoch_pipe in train_pipe.iter_epochs():
        model.train()
        losses = []
        for batch in epoch_pipe.iter_torch_batches(batch_size=batch_size):
            loss, score, label = model(batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        model.eval()
        batch = next(next(test_dataset_iter).iter_torch_batches(batch_size=1000))
        loss, score, label = model(batch)
        test_scores = [score]
        test_labels = [label]

        session.report(
            {
                model.metric_name(): model.compute_metric(
                    test_scores, test_labels
                ).item(),
                "loss": np.mean(losses),
            },
        )


if __name__ == "__main__":
    from azureml.core import Workspace
    from ray_on_aml.core import Ray_On_AML

    ws = Workspace.from_config("docs/advanced/config.json")
    ray_on_aml = Ray_On_AML(ws=ws, compute_cluster="multi-node", maxnode=2)

    ray.init()

    ray = ray_on_aml.getRay(ci_is_head=True, num_node=1)

    trainer = TorchTrainer(
        train_func,
        train_loop_config={
            "num_epochs": 180,
            "feature_idx": 0,
            "feature_dim": 1433,
            "label_idx": 1,
            "label_dim": 1,
            "num_classes": 7,
        },
        run_config=RunConfig(),
        scaling_config=ScalingConfig(num_workers=1),
    )
    result = trainer.fit()

    ray_on_aml.shutdown()
