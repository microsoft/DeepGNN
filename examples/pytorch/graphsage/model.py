# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Collection of graphsage based models."""
from typing import Optional, Tuple

import torch
import numpy as np
import torch.nn as nn

from deepgnn.graph_engine import Graph, FeatureType
from deepgnn.pytorch.common import MeanAggregator, BaseMetric, MRR
from deepgnn.pytorch.modeling import BaseSupervisedModel, BaseUnsupervisedModel
from deepgnn.pytorch.encoding import FeatureEncoder, SageEncoder


class SupervisedGraphSage(BaseSupervisedModel):
    """Simple supervised GraphSAGE model."""

    def __init__(
        self,
        num_classes: int,
        label_idx: int,
        label_dim: int,
        feature_type: FeatureType,
        feature_idx: int,
        feature_dim: int,
        edge_type: int,
        fanouts: list,
        embed_dim: int = 128,
        metric: BaseMetric = MRR(),
        feature_enc: Optional[FeatureEncoder] = None,
    ):
        """Initialize a graphsage model for node classification."""
        super(SupervisedGraphSage, self).__init__(
            feature_type=feature_type,
            feature_idx=feature_idx,
            feature_dim=feature_dim,
            feature_enc=feature_enc,
        )

        # only 1 or 2 hops are allowed.
        assert len(fanouts) in [1, 2]
        self.fanouts = fanouts
        self.edge_type = edge_type

        def feature_func(features):
            return features.squeeze(0)

        first_layer_enc = SageEncoder(
            features=feature_func,
            query_func=None,
            feature_dim=self.feature_dim,
            intermediate_dim=self.feature_enc.embed_dim
            if self.feature_enc
            else self.feature_dim,
            aggregator=MeanAggregator(feature_func),
            embed_dim=embed_dim,
            edge_type=self.edge_type,
            num_sample=self.fanouts[0],
        )

        self.enc = (
            SageEncoder(
                features=lambda context: first_layer_enc(context),
                query_func=first_layer_enc.query,
                feature_dim=self.feature_dim,
                intermediate_dim=embed_dim,
                aggregator=MeanAggregator(lambda context: first_layer_enc(context)),
                embed_dim=embed_dim,
                edge_type=self.edge_type,
                num_sample=self.fanouts[1],
                base_model=first_layer_enc,
            )
            if len(self.fanouts) == 2
            else first_layer_enc
        )

        self.label_idx = label_idx
        self.label_dim = label_dim
        self.weight = nn.Parameter(
            torch.empty(embed_dim, num_classes, dtype=torch.float32)
        )
        self.metric = metric
        nn.init.xavier_uniform_(self.weight)

    def query(self, graph: Graph, inputs: np.ndarray) -> dict:
        """Fetch training data from graph."""
        context = {"inputs": inputs}
        context["label"] = graph.node_features(
            context["inputs"],
            np.array([[self.label_idx, self.label_dim]]),
            FeatureType.INT64,
        )
        context["encoder"] = self.enc.query(
            context["inputs"],
            graph,
            self.feature_type,
            self.feature_idx,
            self.feature_dim,
        )
        self.transform(context)
        return context

    def get_score(self, context: dict) -> torch.Tensor:  # type: ignore[override]
        """Generate scores for a list of nodes."""
        self.encode_feature(context)
        embeds = self.enc(context["encoder"])
        scores = torch.matmul(embeds, self.weight)

        return scores

    def metric_name(self):
        """Metric used for model evaluation."""
        return self.metric.name()

    def get_embedding(self, context: dict) -> torch.Tensor:  # type: ignore[override]
        """Generate embedding."""
        return self.enc(context["encoder"])


class UnSupervisedGraphSage(BaseUnsupervisedModel):
    """Simple unsupervised GraphSAGE model."""

    def __init__(
        self,
        num_classes: int,
        edge_type: int,
        fanouts: list,
        feature_type: FeatureType,
        feature_idx: int,
        feature_dim: int,
        num_negs: int = 20,
        neg_type: int = 0,
        embed_dim: int = 128,
        metric: BaseMetric = MRR(),
        feature_enc: Optional[FeatureEncoder] = None,
    ):
        """
        Initialize graphsage model for unsupervised node classification.

        num_classes -- number of classes of nodes to make predictions.
        enc -- encoder for nodes.
        num_negs -- number of negative samples to use per batch.
        neg_type -- type of nodes for negative samples.
        """
        super(UnSupervisedGraphSage, self).__init__(
            feature_type=feature_type,
            feature_idx=feature_idx,
            feature_dim=feature_dim,
            feature_enc=feature_enc,
        )

        # only 1 or 2 hops are allowed.
        assert len(fanouts) in [1, 2]
        self.num_negs = num_negs
        self.neg_type = neg_type
        self.metric = metric
        self.edge_type = edge_type
        self.fanouts = fanouts

        def feature_func(features):
            return features.squeeze(0)

        first_layer_enc = SageEncoder(
            features=feature_func,
            query_func=None,
            feature_dim=self.feature_dim,
            intermediate_dim=self.feature_enc.embed_dim
            if self.feature_enc
            else self.feature_dim,
            aggregator=MeanAggregator(feature_func),
            embed_dim=embed_dim,
            edge_type=self.edge_type,
            num_sample=self.fanouts[0],
        )

        self.enc = (
            SageEncoder(
                features=lambda context: first_layer_enc(context),
                query_func=first_layer_enc.query,
                feature_dim=self.feature_dim,
                intermediate_dim=embed_dim,
                aggregator=MeanAggregator(lambda context: first_layer_enc(context)),
                embed_dim=embed_dim,
                edge_type=self.edge_type,
                num_sample=self.fanouts[1],
                base_model=first_layer_enc,
            )
            if len(self.fanouts) == 2
            else first_layer_enc
        )
        self.weight = nn.Parameter(
            torch.empty(embed_dim, num_classes, dtype=torch.float32)
        )
        nn.init.xavier_uniform_(self.weight)
        self.bce_loss = torch.nn.BCEWithLogitsLoss()

    def query(self, graph: Graph, inputs: np.ndarray) -> dict:
        """Fetch training data from graph."""
        context = {"inputs": inputs}
        context["encoder"] = self.enc.query(
            context["inputs"],
            graph,
            self.feature_type,
            self.feature_idx,
            self.feature_dim,
        )

        neg = self.get_neg_node(graph, self.num_negs, self.neg_type)
        nbs = self.get_pos_node(
            graph, context["inputs"], np.array([self.edge_type], dtype=np.int32)
        ).flatten()
        context["encoder_neg"] = self.enc.query(
            neg, graph, self.feature_type, self.feature_idx, self.feature_dim
        )
        context["encoder_pos"] = self.enc.query(
            nbs, graph, self.feature_type, self.feature_idx, self.feature_dim
        )
        self.transform(context)
        return context

    def get_embedding(self, context: dict) -> torch.Tensor:  # type: ignore[override]
        """Generate embedding."""
        return self.enc(context["encoder"])

    def get_score(self, context: dict) -> torch.Tensor:  # type: ignore[override]
        """Generate predictions for the list of nodes."""
        self.encode_feature(context)
        embeds = self.enc(context)
        scores = torch.matmul(embeds, self.weight)
        return scores

    def forward(self, context: dict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:  # type: ignore[override]
        """Loss for list of nodes using binary cross entropy."""
        scores = self.get_score(context["encoder"])
        neg_embed = self.get_score(context["encoder_neg"])
        pos_embed = self.get_score(context["encoder_pos"])

        nodes = context["inputs"].squeeze()
        feature_dim = scores.shape[-1]
        batch_size = len(nodes)

        # Prepare embeddings with following dimensions:
        # first is minibatch,
        # second is positive/negative samples for a particular example
        # third is the sample embedding
        embed = scores.reshape(batch_size, 1, feature_dim)
        pos_embed_prepared = pos_embed.reshape(batch_size, 1, feature_dim)
        neg_embed_prepared = neg_embed.expand(batch_size, self.num_negs, feature_dim)

        # Matmul will strengthen matching positions in embeddings for positive examples
        # and keep negative embeddings close to zeros.
        logits = torch.matmul(embed, torch.transpose(pos_embed_prepared, -1, -2))
        neg_logits = torch.matmul(embed, torch.transpose(neg_embed_prepared, -1, -2))

        loss = self.bce_loss(
            torch.cat((logits, neg_logits), dim=2),
            torch.cat((torch.ones_like(logits), torch.zeros_like(neg_logits)), dim=2),
        )

        # Order is important for metrics: mrr is typically calculated based on an assumption
        # the last element is the only positive sample and everything before is negative.
        scores = torch.cat((neg_logits, logits), dim=2)
        labels = torch.cat(
            (torch.zeros_like(neg_logits), torch.ones_like(logits)), dim=2
        )

        return loss, scores, labels

    def metric_name(self):
        """Metric used for model evaluation."""
        return self.metric.name()
