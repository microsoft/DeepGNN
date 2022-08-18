# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Unsupervised GraphSAGE implementation."""
import numpy as np
import tensorflow as tf
from dataclasses import dataclass
from typing import List

from deepgnn.tf.nn import sage_conv
from deepgnn import get_logger

from deepgnn.graph_engine import Graph, SamplingStrategy

from sage import SAGEQuery, SAGEQueryParameter  # type: ignore


@dataclass
class UnsupervisedSamplingParam:
    """Graph query parameters."""

    positive_edge_types: np.ndarray
    positive_sampling_strategy: str = "random"

    negative_node_types: np.ndarray = np.array([0], np.int32)
    negative_num: int = 5


class UnsupervisedQuery(SAGEQuery):
    """GraphSAGE Unsupervised Query."""

    def __init__(
        self, param: SAGEQueryParameter, unsupervised_param: UnsupervisedSamplingParam
    ):
        """Initialize query."""
        super().__init__(param)
        self.unsup_param = unsupervised_param

    def _merge_nodes(self, seed_nodes, positive, negative):
        merged_nodes = np.concatenate(
            [seed_nodes.reshape(-1), positive.reshape(-1), negative.reshape(-1)]
        )
        src_idx = np.arange(0, seed_nodes.size)
        pos_idx = np.arange(seed_nodes.size, seed_nodes.size + positive.size)
        neg_idx = np.arange(seed_nodes.size + positive.size, merged_nodes.size)

        return merged_nodes, src_idx, pos_idx, neg_idx

    def _sample_postive_negative_nodes(self, graph, seed_nodes):
        nbs, w, types, _ = graph.sample_neighbors(
            nodes=seed_nodes,
            edge_types=self.unsup_param.positive_edge_types,
            count=1,
            strategy=self.unsup_param.positive_sampling_strategy,
            default_node=-1,
        )
        positive = nbs
        negative, _ = graph.sample_nodes(
            size=self.unsup_param.negative_num,
            node_types=self.unsup_param.negative_node_types,
            strategy=SamplingStrategy.Weighted,
        )
        return positive, negative

    def query_training(
        self, graph: Graph, inputs: np.ndarray, return_shape: bool = False
    ) -> tuple:
        """Retrieve graph data to train model."""
        seed_nodes = inputs
        positive, negative = self._sample_postive_negative_nodes(graph, seed_nodes)
        merged_nodes, src_idx, pos_idx, neg_idx = self._merge_nodes(
            seed_nodes, positive, negative
        )

        all_nodes, neighbor_list_idx = self._query_neighbor(graph, merged_nodes)

        if self.param.identity_feature:
            graph_tensor = tuple(
                [all_nodes, src_idx, pos_idx, neg_idx] + neighbor_list_idx
            )
        else:
            feat = graph.node_features(
                all_nodes, self.feat_meta, self.param.feature_type
            )
            graph_tensor = tuple(
                [all_nodes, feat, src_idx, pos_idx, neg_idx] + neighbor_list_idx
            )

        # fmt: off
        if return_shape:
            N = None
            if self.param.identity_feature:
                shapes = [
                    [N],                                    # all_nodes
                    list(src_idx.shape),                    # src_idx
                    list(pos_idx.shape),                    # pos_idx
                    list(neg_idx.shape),                    # neg_idx
                ]
            else:
                shapes = [
                    [N],                                    # all_nodes
                    [N, self.param.feature_dim],            # feat
                    list(src_idx.shape),                    # src_idx
                    list(pos_idx.shape),                    # pos_idx
                    list(neg_idx.shape),                    # neg_idx
                ]
            nb_shapes = [list(x.shape) for x in neighbor_list_idx]
            shapes.extend(nb_shapes)
            return graph_tensor, shapes

        return graph_tensor
        # fmt: on


class UnsupervisedGraphSAGE(tf.keras.Model):
    """Unsupervised GraphSAGE model."""

    def __init__(
        self,
        in_dim,
        layer_dims: List[int],
        num_samples: List[int],
        dropout: float = 0.0,
        loss_name: str = "xent",
        agg_type: str = "mean",
        weight_decay: float = 0.0,
        negative_sample_weight: float = 1.0,
        identity_embed_shape: List[int] = [],
        concat: bool = True,
    ):
        """Initialize model."""
        super().__init__()
        self.logger = get_logger()
        # fmt: off
        assert len(layer_dims) == len(num_samples), f"layer_dim {layer_dims}, num_samplers {num_samples}"
        # fmt: on
        self.dims = [in_dim] + layer_dims
        self.num_samples = num_samples
        self.dropout = dropout
        self.loss_name = loss_name
        self.weight_decay = weight_decay
        self.negative_sample_weight = negative_sample_weight
        self.identity_embed_shape = identity_embed_shape  # [num_nodes + 1, dim]
        self.concat = concat

        self.aggs = sage_conv.init_aggregators(
            agg_type, layer_dims, dropout, self.concat
        )

        # use identity features
        if self.identity_embed_shape is not None:
            self.max_id = identity_embed_shape[0] - 1
            with tf.name_scope("op"):
                self.node_emb = tf.Variable(
                    tf.initializers.GlorotUniform()(shape=self.identity_embed_shape),
                    name="node_emb",
                    dtype=tf.float32,
                    trainable=True,
                )

    def _affinity(self, inputs1, inputs2):
        result = tf.reduce_sum(inputs1 * inputs2, axis=1)
        return result

    def _neg_cost(self, inputs1, neg_samples):
        neg_aff = tf.matmul(inputs1, tf.transpose(neg_samples))
        return neg_aff

    def call(self, inputs, training=True):
        """Compute embeddings, loss and MRR metric."""
        # fmt: off
        if self.identity_embed_shape is None:
            nodes, feat, src_idx, pos_idx, neg_idx = inputs[0:5]
            neighbor_list = inputs[5:]
        else:
            nodes, src_idx, pos_idx, neg_idx = inputs[0:4]
            neighbor_list = inputs[4:]
            nodes_filtered = tf.where(nodes >= 0, nodes, tf.ones_like(nodes) * self.max_id)
            feat = tf.nn.embedding_lookup(self.node_emb, tf.reshape(nodes_filtered, [-1]))
        # fmt: on

        hidden = [tf.nn.embedding_lookup(feat, nb) for nb in neighbor_list]
        output = sage_conv.aggregate(
            hidden, self.aggs, self.num_samples, self.dims, self.concat
        )
        output = tf.nn.l2_normalize(output, 1)
        src_emb = tf.nn.embedding_lookup(output, src_idx)
        pos_emb = tf.nn.embedding_lookup(output, pos_idx)
        neg_emb = tf.nn.embedding_lookup(output, neg_idx)
        self.logger.info(
            f"src {src_emb.shape}, pos {pos_emb.shape}, neg {neg_emb.shape}"
        )

        # embedding result
        src_nodes_idx = tf.nn.embedding_lookup(neighbor_list[0], src_idx)
        self.src_nodes = tf.nn.embedding_lookup(nodes, src_nodes_idx)
        self.src_emb = src_emb

        # MRR
        mrr = self._calc_mrr(src_emb, pos_emb, neg_emb)

        # Loss
        loss = self._xent_loss(src_emb, pos_emb, neg_emb)
        batch_size = tf.cast(tf.shape(src_emb)[0], tf.float32)
        loss = loss / batch_size

        return src_emb, loss, {"mrr": mrr}

    def _calc_mrr(self, src_emb, pos_emb, neg_emb):
        # src_emb, pos_emb, neg_emb
        # [N, D],  [N, D], [#neg, D]
        # fmt: off
        aff = self._affinity(src_emb, pos_emb)                   # [N]
        neg_aff = self._neg_cost(src_emb, neg_emb)               # [N, #neg]
        aff = tf.reshape(aff, [-1, 1])                          # [N, 1]
        aff_all = tf.concat(axis=1, values=[neg_aff, aff])      # [N, #neg+1]
        self.logger.info(f"aff {aff.shape}, neg_aff {neg_aff.shape}, all {aff_all.shape}")
        size = tf.shape(aff_all)[1]
        _, indices_of_ranks = tf.nn.top_k(aff_all, k=size)
        _, ranks = tf.nn.top_k(-indices_of_ranks, k=size)
        mrr = tf.reduce_mean(
            input_tensor=tf.math.reciprocal(tf.cast(ranks[:, -1] + 1, dtype=tf.float32))
        )
        # fmt: on
        return mrr

    def _xent_loss(self, inputs1, inputs2, neg_samples):
        aff = self._affinity(inputs1, inputs2)
        neg_aff = self._neg_cost(inputs1, neg_samples)
        true_xent = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.ones_like(aff), logits=aff
        )
        negative_xent = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.zeros_like(neg_aff), logits=neg_aff
        )
        loss = tf.reduce_sum(true_xent) + self.negative_sample_weight * tf.reduce_sum(
            negative_xent
        )
        return loss

    def train_step(self, data: dict):
        """Override base train_step."""
        with tf.GradientTape() as tape:
            _, loss, metrics = self(data, training=True)

        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        result = {"loss": loss}
        result.update(metrics)
        return result

    def test_step(self, data: dict):
        """Override base test_step."""
        _, loss, metrics = self(data, training=False)
        result = {"loss": loss}
        result.update(metrics)
        return result

    def predict_step(self, data: dict):
        """Override base predict_step."""
        self(data, training=False)
        return [self.src_nodes, self.src_emb]
