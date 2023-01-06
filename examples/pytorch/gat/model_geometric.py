# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""GAT model implementation with torch geometric."""
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn.functional as F
from typing import List

from deepgnn.pytorch.common import Accuracy
from deepgnn.pytorch.modeling.base_model import BaseModel

from deepgnn.graph_engine import Graph, graph_ops
from torch_geometric.nn import GATConv


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

        graph_tensor = {"nodes": np.expand_dims(nodes, 0), "feat": np.expand_dims(feat, 0), "edge_index": np.expand_dims(edges, 0), "mask": np.expand_dims(input_mask, 0), "label": np.expand_dims(label, 0)}
        return graph_tensor


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
        # fmt: off
        nodes = torch.squeeze(inputs["nodes"])                # [N]
        feat = torch.squeeze(inputs["feat"])                  # [N, F]
        edge_index = torch.squeeze(inputs["edge_index"])      # [2, X]
        mask = torch.squeeze(inputs["mask"])                  # [N]
        labels = torch.squeeze(inputs["label"])               # [N]
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
