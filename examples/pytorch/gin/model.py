from multiprocessing import pool
from multiprocessing.sharedctypes import Value
from optparse import Option
from typing import Optional

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from deepgnn.graph_engine import Graph, FeatureType
from deepgnn.pytorch.common import MeanAggregator, BaseMetric, MRR
from deepgnn.pytorch.modeling import BaseSupervisedModel
from deepgnn.pytorch.encoding import FeatureEncoder
from deepgnn import get_logger


class MLP(nn.Module):
    """Simple MLP with linear-output model."""

    def __init__(
        self,
        num_layers: int,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
    ):
        super(MLP, self).__init__()
        self.num_layers = num_layers

        if num_layers < 1:
            raise ValueError("Num layers should be > 0")
        elif num_layers == 1:
            # Single linear model
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            # Multi-layer model
            self.linears = torch.nn.ModuleList()
            self.batch_norms = torch.nn.ModuleList()

            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))

            self.linears.append(nn.Linear(hidden_dim, output_dim))

            for layer in range(num_layers - 1):
                self.batch_norms.append(nn.BatchNorm1d((hidden_dim)))

    def forward(self, x):
        if self.num_layers == 1:
            return self.linear(x)
        else:
            h = x
            for layer in range(self.num_layers - 1):
                h = F.relu(self.batch_norms[layer][self.linears[layer](h)])
            return self.linears[self.num_layers - 1](h)


class GIN(BaseSupervisedModel):
    """Simple supervised GIN model."""

    """
            num_layers: number of layers in the neural networks (INCLUDING the input layer)
            num_mlp_layers: number of layers in mlps (EXCLUDING the input layer)
            input_dim: dimensionality of input features
            hidden_dim: dimensionality of hidden units at ALL layers
            output_dim: number of classes for prediction
            final_dropout: dropout ratio on the final linear layer
            learn_eps: If True, learn epsilon to distinguish center nodes from neighboring nodes. If False, aggregate neighbors and center nodes altogether.
            neighbor_pooling_type: how to aggregate neighbors (mean, average, or max)
            graph_pooling_type: how to aggregate entire nodes in a graph (mean, average)
            device: which device to use
        """

    def __init__(
        self,
        num_layers: int,
        num_mlp_layers: int,
        edge_type: np.ndarray,
        input_dim: int,
        label_idx: int,
        label_dim: int,
        hidden_dim: int,
        output_dim: int,
        feature_type: FeatureType,
        feature_idx: int,
        feature_dim: int,
        final_dropout: str,
        learn_eps: bool,
        graph_pooling_type: str,
        neighbor_pooling_type: str,
        device: None,
        metric: BaseMetric = MRR(),
        feature_enc: Optional[FeatureEncoder] = None,
    ):

        super(GIN, self).__init__(
            feature_type=feature_type,
            feature_idx=feature_idx,
            feature_dim=feature_dim,
            feature_enc=feature_enc,
        )

        self.final_dropout = final_dropout
        self.device = device
        self.metric = metric
        self.edge_type = edge_type
        self.num_layers = num_layers
        self.graph_pooling_type = graph_pooling_type
        self.neighbor_pooling_type = neighbor_pooling_type
        self.learn_eps = learn_eps
        self.eps = nn.Parameter(torch.zeros(self.num_layers - 1))

        self.label_idx = label_idx
        self.label_dim = label_dim
        self.feature_type = feature_type
        self.feature_idx = feature_idx
        self.feature_dim = feature_dim

        ###List of MLPs
        self.mlps = torch.nn.ModuleList()

        ###List of batchnorms applied to the output of MLP (input of the final prediction linear layer)
        self.batch_norms = torch.nn.ModuleList()

        for layer in range(self.num_layers - 1):
            if layer == 0:
                self.mlps.append(MLP(num_mlp_layers, input_dim, hidden_dim, hidden_dim))
            else:
                self.mlps.append(
                    MLP(num_mlp_layers, hidden_dim, hidden_dim, hidden_dim)
                )

            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        # Linear function that maps the hidden representation at dofferemt layers into a prediction score
        self.linears_prediction = torch.nn.ModuleList()
        for layer in range(num_layers):
            if layer == 0:
                self.linears_prediction.append(nn.Linear(input_dim, output_dim))
            else:
                self.linears_prediction.append(nn.Linear(hidden_dim, output_dim))

    def next_layer(self, h, layer, context):

        # Sum Pooling
        # for i in range(len())
        pooled = torch.zeros(0)
        for i in range(len(context["nb_counts"][0])):
            num_neighbors = context["nb_counts"][0][i].int()
            sum_features = torch.sum(context["features"][0][num_neighbors])

        # Average Pooling
        if self.neighbor_pooling_type == "average":
            degree = sum(context["label"])
            pooled = pooled / degree

        # representation of neighboring and center nodes
        pooled_rep = self.mlps[layer](pooled)
        h = self.batch_norms[layer](pooled_rep)
        h = F.relu(h)
        return h

    def get_score(self, context: dict):
        num_nodes = len(context["nb_counts"][0])
        offset = 0

        # Read from context, condense vector
        nb_counts = context["nb_counts"].squeeze()
        features = context["features"].squeeze()

        # Allocate memory for pooling matrix
        pooled_h = torch.zeros(num_nodes, self.feature_dim)

        # Loop across all nodes in mini-batch
        for node_id in range(num_nodes):
            num_neighbors = nb_counts[node_id].int().item()

            # Aggregate and sum features across all neighbors
            neighbor_features = features[offset : offset + num_neighbors]
            sum_features = neighbor_features.sum(0)

            # Move offset forward to read next set of nb features
            offset += num_neighbors + 1

            # Write to pooled matrix
            pooled_h[node_id] = sum_features

        score = 0
        for layer in range(self.num_layers):
            score += F.dropout(
                self.linears_prediction[layer](pooled_h),
                self.final_dropout,
                training=self.training,
            )

        return score

    def forward(self, context: dict):
        scores: torch.Tensor = self.get_score(context)
        labels = context["label"].squeeze().long().clone().detach()

        # Calculate cross-entropy loss
        loss = self.xent(scores, labels)

        # Take argmax to fetch class indices
        scores = scores.argmax(dim=1)

        return (loss, scores, labels)

    def metric_name(self):
        """Metric used for model evaluation."""
        return self.metric.name()

    def query(self, graph: Graph, inputs: np.array):
        """Fetch training data from graph."""
        context = {"inputs": inputs}

        # context['neighbors'] = graph.neighbors(
        #     context["inputs"],
        #     np.array([0])
        # )[0]

        context["neighbors"] = graph.sample_neighbors(
            nodes=context["inputs"],
            edge_types=np.array([0, 1, 2, 3]),
            count=10,
            strategy="randomwithoutreplacement",
        )[0]

        context["nb_counts"] = graph.neighbor_count(
            nodes=context["inputs"],
            edge_types=np.array([0]),
        ).astype(float)

        context["label"] = graph.node_features(
            context["inputs"],
            np.array([[self.label_idx, self.label_dim]]),
            FeatureType.FLOAT,
        )

        context["features"] = graph.node_features(
            context["neighbors"],
            np.array([[self.label_idx, self.label_dim]]),
            FeatureType.FLOAT,
        )

        return context
