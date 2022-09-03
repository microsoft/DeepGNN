# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""GAT model implementation."""
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn.functional as F
from typing import List

from deepgnn.pytorch.common import Accuracy
from deepgnn.pytorch.modeling.base_model import BaseModel
from deepgnn.pytorch.nn.gat_conv import GATConv

from deepgnn.graph_engine import Graph, FeatureType, graph_ops


@dataclass
class GATQueryParameter:
    """Graph query configuration for GAT model."""

    neighbor_edge_types: np.ndarray
    feature_idx: int
    feature_dim: int
    label_idx: int
    label_dim: int
    feature_type: FeatureType = FeatureType.FLOAT
    label_type: FeatureType = FeatureType.FLOAT
    num_hops: int = 2


class GATQuery:
    """Graph query to generate data for GAT model."""

    def __init__(self, p: GATQueryParameter):
        """Initialize graph query."""
        self.p = p
        self.label_meta = np.array([[p.label_idx, p.label_dim]], np.int32)
        self.feat_meta = np.array([[p.feature_idx, p.feature_dim]], np.int32)

    def query_training(self, graph: Graph, inputs: np.ndarray) -> tuple:
        """Query used to generate data for training."""
        nodes, edges, src_idx = graph_ops.sub_graph(
            graph,
            inputs,
            edge_types=self.p.neighbor_edge_types,
            num_hops=self.p.num_hops,
            self_loop=True,
            undirected=True,
            return_edges=True,
        )
        input_mask = np.zeros(nodes.size, np.bool)
        input_mask[src_idx] = True

        feat = graph.node_features(nodes, self.feat_meta, self.p.feature_type)
        label = graph.node_features(nodes, self.label_meta, self.p.label_type)
        label = label.astype(np.int32)
        edges_value = np.ones(edges.shape[0], np.float32)
        edges = np.transpose(edges)
        adj_shape = np.array([nodes.size, nodes.size], np.int64)

        graph_tensor = (nodes, feat, input_mask, label, edges, edges_value, adj_shape)
        return graph_tensor


class GAT(BaseModel):
    """GAT model implementation."""

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
        """Initialize GAT model."""
        self.q = GATQuery(q_param)
        super().__init__(FeatureType.FLOAT, 0, 0, None)
        self.num_classes = num_classes

        self.out_dim = num_classes

        self.input_layer = GATConv(
            in_dim=in_dim,
            attn_heads=head_num[0],
            out_dim=hidden_dim,
            act=F.elu,
            in_drop=ffd_drop,
            coef_drop=attn_drop,
            attn_aggregate="concat",
        )
        layer0_output_dim = head_num[0] * hidden_dim
        # TODO: support hidden layer
        assert len(head_num) == 2
        self.out_layer = GATConv(
            in_dim=layer0_output_dim,
            attn_heads=head_num[1],
            out_dim=self.out_dim,
            act=None,
            in_drop=ffd_drop,
            coef_drop=attn_drop,
            attn_aggregate="average",
        )

        self.metric = Accuracy()

    def forward(self, inputs):
        """Evaluate model, calculate loss, predictions and extract labels."""
        # fmt: off
        nodes, feat, mask, labels, edges, edges_value, adj_shape = inputs
        nodes = torch.squeeze(nodes)                # [N], N: num of nodes in subgraph
        feat = torch.squeeze(feat)                  # [N, F]
        mask = torch.squeeze(mask)                  # [N]
        labels = torch.squeeze(labels)              # [N]
        edges = torch.squeeze(edges)                # [X, 2], X: num of edges in subgraph
        edges_value = torch.squeeze(edges_value)    # [X]
        adj_shape = torch.squeeze(adj_shape)        # [2]
        # fmt: on

        sp_adj = torch.sparse_coo_tensor(edges, edges_value, adj_shape.tolist())
        h_1 = self.input_layer(feat, sp_adj)
        scores = self.out_layer(h_1, sp_adj)

        labels = labels.type(torch.int64)
        labels = labels[mask]  # [batch_size]
        scores = scores[mask]  # [batch_size]
        pred = scores.argmax(dim=1)
        loss = self.xent(scores, labels)
        return loss, pred, labels
