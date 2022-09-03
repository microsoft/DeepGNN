# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""HetGnn model implementation."""

from typing import List, Any, Dict, Union, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from deepgnn.graph_engine import Graph, FeatureType
from deepgnn.pytorch.common import MRR
from deepgnn.pytorch.modeling import BaseUnsupervisedModel


class HetGnnModel(BaseUnsupervisedModel):
    """Core heterogenous gnn model."""

    def __init__(
        self,
        node_type_count: int,
        neighbor_count: int,
        embed_d: int,
        feature_idx: int,
        feature_dim: int,
        feature_type: FeatureType,
        metric=MRR(),
    ):
        """Initialize HetGnn model."""
        super(HetGnnModel, self).__init__(
            feature_type=feature_type,
            feature_idx=feature_idx,
            feature_dim=feature_dim,
            feature_enc=None,
        )

        self.embed_d = embed_d
        self.node_type_count = node_type_count
        self.neighbor_count = neighbor_count
        self.metric = metric

        # NOTE: do not use python list to store nn layers, use nn.ModuleList, otherwise
        # the parameters will not added to the optimizer by default and loss function won't work.
        self.content_rnn = nn.ModuleList(
            [
                nn.LSTM(embed_d, int(embed_d / 2), 1, bidirectional=True)
                for i in range(node_type_count)
            ]
        )
        self.neigh_rnn = nn.ModuleList(
            [
                nn.LSTM(embed_d, int(embed_d / 2), 1, bidirectional=True)
                for i in range(node_type_count)
            ]
        )
        self.neigh_att = nn.ParameterList(
            [
                nn.Parameter(torch.ones(embed_d * 2, 1), requires_grad=True)
                for i in range(node_type_count)
            ]
        )

        self.softmax = nn.Softmax(dim=1)
        self.act = nn.LeakyReLU()
        self.drop = nn.Dropout(p=0.5)
        self.bn = nn.BatchNorm1d(embed_d)

        self.init_weights()

    def init_weights(self):
        """Set internal weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Parameter):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)

    def get_score(self, context: dict) -> torch.Tensor:  # type: ignore[override]
        """Calculate scores for central, positive and negative nodes."""
        c_out, p_out, n_out = self.aggregate_all(context)
        return c_out, p_out, n_out

    def aggregate_all(self, context: dict):
        """Split triples into: central, positive and negative node lists."""
        c_agg, pos_agg, neg_agg = self.het_agg(context)
        return c_agg, pos_agg, neg_agg

    def het_agg(self, context: dict):
        """Aggregate all neighbor nodes."""
        triple_index = int(context["triple_index"])
        # compute the node type based on the triple index.
        context["c"]["node_type"] = triple_index // self.node_type_count
        context["p"]["node_type"] = triple_index % self.node_type_count
        context["n"]["node_type"] = triple_index % self.node_type_count

        c_agg = self.node_het_agg(context["c"])
        p_agg = self.node_het_agg(context["p"])
        n_agg = self.node_het_agg(context["n"])

        return c_agg, p_agg, n_agg

    def node_het_agg(self, context: dict):
        """
        Aggregate neighbor nodes based on node type.

        This will call content aggregator(NN-1) inside.
        """
        agg_batch: List[Any] = [[]] * self.node_type_count
        for i in range(self.node_type_count):
            agg_batch[i] = self.node_neigh_agg(i, context["neigh_feats"][i])

        # attention module
        node_type = int(context["node_type"])
        c_agg_batch = self.content_agg(node_type, context["node_feats"])
        c_agg_batch_2 = torch.cat((c_agg_batch, c_agg_batch), dim=1).view(
            len(c_agg_batch), self.embed_d * 2
        )
        agg_batch_2 = [
            torch.cat((c_agg_batch, agg_batch[i]), dim=1).view(
                len(c_agg_batch), self.embed_d * 2
            )
            for i in range(len(agg_batch))
        ]

        # compute weights
        agg_batch_2.insert(0, c_agg_batch_2)
        concate_embed = torch.cat(agg_batch_2, 1).view(
            len(c_agg_batch), self.node_type_count + 1, self.embed_d * 2
        )
        atten_w = self.act(
            torch.bmm(
                concate_embed,
                self.neigh_att[node_type]
                .unsqueeze(0)
                .expand(len(c_agg_batch), *self.neigh_att[node_type].size()),
            )
        )
        atten_w = self.softmax(atten_w).view(
            len(c_agg_batch), 1, self.node_type_count + 1
        )

        # weighted combination
        agg_batch.insert(0, c_agg_batch)
        concate_embed = torch.cat(agg_batch, 1).view(
            len(c_agg_batch), self.node_type_count + 1, self.embed_d
        )
        weight_agg_batch = torch.bmm(atten_w, concate_embed).view(
            len(c_agg_batch), self.embed_d
        )

        return weight_agg_batch

    def node_neigh_agg(
        self, node_type: int, neigh_feats: torch.Tensor
    ) -> torch.Tensor:  # type based neighbor aggregation with rnn
        """
        Heterogeneous gnn NN-2 implementation.

        Aggregating all the neighbors of node grouped by node types
        using BiLSTM and mean pooling.
        """
        batch_s = int(neigh_feats.squeeze(0).shape[0] / self.neighbor_count)

        neigh_agg = self.content_agg(node_type, neigh_feats).view(
            batch_s, self.neighbor_count, self.embed_d
        )
        neigh_agg = torch.transpose(neigh_agg, 0, 1)

        all_state, _ = self.neigh_rnn[node_type](neigh_agg)
        neigh_agg = torch.mean(all_state, 0).view(batch_s, self.embed_d)
        return neigh_agg

    def content_agg(
        self, node_type: int, feats: torch.Tensor
    ) -> torch.Tensor:  # heterogeneous content aggregation
        """
        Heterogeneous gnn NN-1 implement.

        Aggregating all the heterogeneous content of node using BiLSTM and mean pooling.
        """
        feature_list = feats.squeeze(0)  # self.features(torch.as_tensor(id_batch[0]))
        concate_embed = feature_list.view(feature_list.shape[0], 1, self.embed_d)
        concate_embed = torch.transpose(concate_embed, 0, 1)
        all_state, _ = self.content_rnn[node_type](concate_embed)

        return torch.mean(all_state, 0)

    def metric_name(self):
        """Metric used for model evaluation."""
        return self.metric.name()

    def cross_entropy_loss(
        self,
        c_embed_batch: np.ndarray,
        pos_embed_batch: np.ndarray,
        neg_embed_batch: np.ndarray,
        embed_d: int,
    ):
        """Evaluate mean loss for all node types."""
        batch_size = c_embed_batch.shape[0] * c_embed_batch.shape[1]

        c_embed = c_embed_batch.view(batch_size, 1, embed_d)
        pos_embed = pos_embed_batch.view(batch_size, embed_d, 1)
        neg_embed = neg_embed_batch.view(batch_size, embed_d, 1)

        out_p = torch.bmm(c_embed, pos_embed)
        out_n = -torch.bmm(c_embed, neg_embed)

        sum_p = F.logsigmoid(out_p)
        sum_n = F.logsigmoid(out_n)
        loss_sum = -(sum_p + sum_n)

        scores = torch.cat([out_n, out_p], dim=2)
        labels = torch.cat([torch.zeros_like(out_n), torch.ones_like(out_p)], dim=2)

        return loss_sum.mean(), scores, labels

    def forward(self, context: dict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:  # type: ignore[override]
        """Calculate score based on inputs in the context and return loss."""
        feature_list = context["encoder"]
        inputs = context["inputs"]
        size_array = max([len(inputs[k]) for k in range(len(inputs))])
        c_out = torch.zeros([len(feature_list), size_array, self.embed_d])
        p_out = torch.zeros([len(feature_list), size_array, self.embed_d])
        n_out = torch.zeros([len(feature_list), size_array, self.embed_d])

        for feature_index in range(len(feature_list)):
            feature_list_temp = feature_list[feature_index]
            if len(feature_list_temp) == 0:
                continue

            context["encoder"][feature_index]["triple_index"] = feature_index
            c_out_temp, p_out_temp, n_out_temp = self.get_score(
                context["encoder"][feature_index]
            )

            c_out[feature_index][0 : c_out_temp.shape[0]] = c_out_temp
            p_out[feature_index][0 : p_out_temp.shape[0]] = p_out_temp
            n_out[feature_index][0 : n_out_temp.shape[0]] = n_out_temp

        return self.cross_entropy_loss(c_out, p_out, n_out, self.embed_d)

    def get_embedding(self, context: dict) -> torch.Tensor:  # type: ignore[override]
        """Calculate embedding."""
        context["encoder"]["node_type"] = int(context["node_type"])
        return self.node_het_agg(context["encoder"])

    def build_node_context(self, id_batch, graph):
        """Fetch node features from graph."""
        context = {}
        neigh_batch = np.empty(
            (self.node_type_count, len(id_batch), self.neighbor_count), dtype=np.int64
        )
        neigh_batch_feature = []
        context["node_feats"] = graph.node_features(
            id_batch,
            np.array([[self.feature_idx, self.feature_dim]]),
            FeatureType.FLOAT,
        )
        for i in range(self.node_type_count):
            neigh_batch[i] = graph.sample_neighbors(
                id_batch, np.array([i], dtype=np.int32), self.neighbor_count
            )[0]
            neigh_batch_feature.append(
                graph.node_features(
                    np.reshape(neigh_batch[i], (1, -1))[0],
                    np.array([[self.feature_idx, self.feature_dim]]),
                    FeatureType.FLOAT,
                )
            )
        context["neigh_feats"] = neigh_batch_feature

        return context

    def build_triple_context(self, triple_list_batch, graph):
        """Fetch features for all node triples."""
        c_id_batch = triple_list_batch[:, 0]
        pos_id_batch = triple_list_batch[:, 1]
        neg_id_batch = triple_list_batch[:, 2]

        context = {}

        context["c"] = self.build_node_context(c_id_batch, graph)
        context["p"] = self.build_node_context(pos_id_batch, graph)
        context["n"] = self.build_node_context(neg_id_batch, graph)

        return context

    def query(self, graph: Graph, inputs: np.ndarray) -> dict:
        """Query graph for training data."""
        context = {"inputs": inputs}
        triple_context: List[Union[Dict, List]] = []
        for triple_index in range(len(inputs)):
            triple_list_temp = inputs[triple_index]
            if len(triple_list_temp) == 0:
                triple_context.append([])
                continue

            triple_context.append(
                self.build_triple_context(np.array(triple_list_temp, np.int64), graph)
            )
        context["encoder"] = triple_context
        return context

    def query_inference(self, graph: Graph, inputs: np.ndarray) -> dict:
        """Query graph to generate embeddings."""
        context = {}
        context["inputs"] = inputs[:, 0]
        context["node_type"] = inputs[0][1]
        context["encoder"] = self.build_node_context(context["inputs"], graph)
        return context
