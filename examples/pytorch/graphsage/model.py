# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Simple supervised GraphSAGE model implemented with PyTorch geometric."""
from typing import Optional

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn

from deepgnn.graph_engine import Graph
from deepgnn.pytorch.common import BaseMetric, MRR
from deepgnn.pytorch.modeling import BaseSupervisedModel
from deepgnn.pytorch.encoding import FeatureEncoder


class PTGSupervisedGraphSage(BaseSupervisedModel):
    """Supervised graphsage model implementation with torch geometric."""

    def __init__(
        self,
        num_classes: int,
        label_idx: int,
        label_dim: int,
        feature_type: np.dtype,
        feature_idx: int,
        feature_dim: int,
        edge_type: int,
        fanouts: list,
        embed_dim: int = 128,
        metric: BaseMetric = MRR(),
        feature_enc: Optional[FeatureEncoder] = None,
    ):
        """Initialize a graphsage model for node classification."""
        super(PTGSupervisedGraphSage, self).__init__(
            feature_type=feature_type,
            feature_idx=feature_idx,
            feature_dim=feature_dim,
            feature_enc=feature_enc,
        )

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

        context["x0"] = x0
        context["out_1"] = n1_out.shape[0]  # Number of output nodes of layer 1
        context["out_2"] = n2_out.shape[0]  # Number of output nodes of layer 2
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
        out_1 = context["out_1"]
        out_2 = context["out_2"]
        edges_1 = self.build_edges_tensor(out_1, self.fanouts[0])  # Edges for 1st layer
        x1 = self.convs[0](context["x0"].squeeze(), edges_1)[
            :out_1, :
        ]  # Output of 1st layer (cut out 2 hop nodes)
        x1 = F.relu(x1)
        edges_2 = self.build_edges_tensor(out_2, self.fanouts[1])  # Edges for 2nd layer
        x2 = self.convs[1](x1, edges_2)[
            :out_2, :
        ]  # Output of second layer (nodes for which loss is computed)
        x2 = F.relu(x2)
        return x2
