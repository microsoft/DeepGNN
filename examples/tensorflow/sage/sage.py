# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""GraphSAGE models implementations."""
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from dataclasses import dataclass
from typing import List

from deepgnn.tf.nn import sage_conv

from deepgnn.graph_engine import Graph, FeatureType


@dataclass
class LayerInfo:
    """Layer configuration."""

    num_samples: int
    neighbor_edge_types: np.ndarray
    strategy: str


@dataclass
class SAGEQueryParameter:
    """Query configuration for sage models."""

    layer_infos: List[LayerInfo]
    feature_idx: int
    feature_dim: int
    label_idx: int
    label_dim: int
    feature_type: FeatureType = FeatureType.FLOAT
    label_type: FeatureType = FeatureType.FLOAT
    identity_feature: bool = False


class SAGEQuery:
    """GraphSAGE Query."""

    def __init__(self, param: SAGEQueryParameter):
        """Initialize query."""
        self.param = param
        self.label_meta = np.array([[param.label_idx, param.label_dim]], np.int32)
        self.feat_meta = np.array([[param.feature_idx, param.feature_dim]], np.int32)

        assert (self.param.identity_feature and self.param.feature_dim <= 0) or (
            not self.param.identity_feature and self.param.feature_dim > 0
        ), f"feature less {self.param.identity_feature}, feature_dim {self.param.feature_dim}"

    def _sample_n_hop_neighbors(
        self, graph: Graph, inputs: np.ndarray, layer_infos: List[LayerInfo]
    ):
        """
        GraphSAGE: Sample neighbors for multi-layer convolutions.

        Reference:https://github.com/williamleif/GraphSAGE/blob/a0fdef95dca7b456dab01cb35034717c8b6dd017/graphsage/models.py#L254
        """
        neighbor_list = [inputs]
        support_size = 1
        support_sizes = [support_size]
        for k in range(len(layer_infos)):
            t = len(layer_infos) - k - 1
            cur_nodes = neighbor_list[k]
            support_size *= layer_infos[t].num_samples
            neighbors, w, types, _ = graph.sample_neighbors(
                nodes=cur_nodes,
                edge_types=layer_infos[t].neighbor_edge_types,
                count=layer_infos[t].num_samples,
                strategy=layer_infos[t].strategy,
                default_node=-1,
            )
            neighbor_list.append(neighbors.reshape(-1))
            support_sizes.append(support_size)
        return neighbor_list, support_sizes

    def _query_neighbor(self, graph, inputs):
        seed_nodes = inputs
        neighbor_list, support_size = self._sample_n_hop_neighbors(
            graph, seed_nodes, self.param.layer_infos
        )

        all_neighbors = np.concatenate(neighbor_list)
        all_nodes, idx = np.unique(all_neighbors, return_inverse=True)
        neighbor_list_idx = []
        nb_sizes = [x.size for x in neighbor_list]
        offset = 0
        for s in nb_sizes:
            neighbor_list_idx.append(idx[offset : offset + s])
            offset += s

        return all_nodes, neighbor_list_idx

    def query_training(
        self, graph: Graph, inputs: np.ndarray, return_shape: bool = False
    ) -> tuple:
        """Fetch training data from graph."""
        # fmt: off
        seed_nodes = inputs
        all_nodes, neighbor_list_idx = self._query_neighbor(graph, seed_nodes)
        label = graph.node_features(seed_nodes, self.label_meta, self.param.label_type)
        if self.param.identity_feature:
            graph_tensor = tuple([all_nodes, label] + neighbor_list_idx)
        else:
            feat = graph.node_features(all_nodes, self.feat_meta, self.param.feature_type)
            graph_tensor = tuple([all_nodes, feat, label] + neighbor_list_idx)

        if return_shape:
            # N is the number of `nodes`, which is variable because `inputs` nodes are different.
            N = None
            if self.param.identity_feature:
                shapes = [
                    [N],                                        # Nodes
                    [inputs.size, self.param.label_dim],        # label
                ]
            else:
                shapes = [
                    [N],                                        # Nodes
                    [N, self.param.feature_dim],                # feat
                    [inputs.size, self.param.label_dim],        # label
                ]

            nb_shapes = [list(x.shape) for x in neighbor_list_idx]
            shapes.extend(nb_shapes)
            return graph_tensor, shapes

        return graph_tensor
        # fmt: on


class GraphSAGE(tf.keras.Model):
    """Base GraphSAGE model."""

    def __init__(
        self,
        in_dim,
        layer_dims: List[int],
        num_classes: int,
        num_samples: List[int],
        dropout: float = 0.0,
        loss_name: str = "softmax",
        agg_type: str = "mean",
        weight_decay: float = 0.0,
        identity_embed_shape: List[int] = [],
        concat: bool = True,
    ):
        """Initialize model."""
        super().__init__()
        # fmt: off
        assert len(layer_dims) == len(num_samples), f"layer_dim {layer_dims}, num_samplers {num_samples}"
        assert loss_name in ["sigmoid", "softmax"], f"unknown loss name {loss_name}"
        # fmt: on

        self.dims = [in_dim] + layer_dims
        self.num_classes = num_classes
        self.num_samples = num_samples
        self.dropout = dropout
        self.loss_name = loss_name
        self.weight_decay = weight_decay
        self.identity_embed_shape = identity_embed_shape
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

        self.pred_layer = tf.keras.layers.Dense(self.num_classes, name="node_pred")
        self.f1_micro = tfa.metrics.F1Score(num_classes=num_classes, average="micro")
        self.f1_macro = tfa.metrics.F1Score(num_classes=num_classes, average="macro")

    def reset_metric_states(self):
        """Reset metric states."""
        self.f1_macro.reset_states()
        self.f1_micro.reset_states()

    def call(self, inputs, training=True):
        """Return input tensors, loss and metric values."""
        # fmt: off
        if self.identity_embed_shape is None:
            nodes, feat, label = inputs[0:3]
            neighbor_list = inputs[3:]
        else:
            nodes, label = inputs[0:2]
            neighbor_list = inputs[2:]
            nodes_filtered = tf.where(nodes >= 0, nodes, tf.ones_like(nodes) * self.max_id)
            feat = tf.nn.embedding_lookup(self.node_emb, tf.reshape(nodes_filtered, [-1]))
        # fmt: on

        hidden = [tf.nn.embedding_lookup(feat, nb) for nb in neighbor_list]
        output = sage_conv.aggregate(
            hidden, self.aggs, self.num_samples, self.dims, self.concat
        )

        # output layer
        output = tf.nn.l2_normalize(output, 1)

        if self.dropout != 0.0:
            output = tf.nn.dropout(output, rate=self.dropout)
        node_preds = self.pred_layer(output)

        # embedding results
        src_nodes_idx = neighbor_list[0]
        self.src_nodes = tf.nn.embedding_lookup(nodes, src_nodes_idx)
        self.src_emb = node_preds

        if label.shape[1] == 1 and self.num_classes != 1:
            # The label_dim of the cora datasets is 1, and num_classes is not 1.
            # Here is to convert label to one hot vector if label_dim is 1.
            label = tf.cast(label, tf.int32)
            label = tf.one_hot(label, self.num_classes)
            label = tf.reshape(label, [-1, self.num_classes])

        loss, pred = self.calc_loss(node_preds, label)

        # update metrics
        acc = self.calc_accuracy(pred, label)
        self.f1_macro.update_state(label, pred)
        self.f1_micro.update_state(label, pred)

        return (
            nodes,
            loss,
            {
                "f1_macro": self.f1_macro.result(),
                "f1_micro": self.f1_micro.result(),
                "accuracy": acc,
            },
        )

    def calc_accuracy(self, preds, labels):
        """Caluclate prediction accuracy."""
        if self.loss_name == "sigmoid":
            correct_prediction = tf.equal(preds, labels)
        else:
            correct_prediction = tf.equal(tf.argmax(preds, 1), tf.argmax(labels, 1))

        accuracy_all = tf.cast(correct_prediction, tf.float32)
        acc = tf.reduce_mean(accuracy_all)
        return acc

    def calc_loss(self, node_preds, label):
        """Classification loss."""
        if self.loss_name == "sigmoid":
            logits = tf.nn.sigmoid_cross_entropy_with_logits(
                logits=node_preds, labels=label
            )
            pred = tf.nn.sigmoid(node_preds)
            pred = tf.where(pred > 0.5, tf.ones_like(pred), tf.zeros_like(pred))
        else:
            logits = tf.nn.softmax_cross_entropy_with_logits(
                logits=node_preds, labels=label
            )
            pred = tf.nn.softmax(node_preds)
        loss = tf.reduce_mean(logits)

        loss += self._calc_l2_loss()
        return loss, pred

    def _calc_l2_loss(self):
        vs = []
        for v in self.trainable_variables:
            vs.append(tf.nn.l2_loss(v))
        l2_loss = tf.add_n(vs) * self.weight_decay
        return l2_loss

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
