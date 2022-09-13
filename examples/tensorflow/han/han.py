# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""HAN model implementation."""
import numpy as np
import tensorflow as tf
from dataclasses import dataclass
from deepgnn.tf import encoders
from deepgnn.graph_engine import Graph, FeatureType
from deepgnn.graph_engine import multihop
from typing import Tuple


@dataclass
class HANQueryParamemter:
    """Graph query parameters for HAN model."""

    edge_types: np.ndarray
    feature_idx: int
    feature_dim: int
    label_idx: int
    label_dim: int
    nb_num: list
    max_id: int = -1


class HANQuery:
    """Graph Query: get sub graphs for HAN training."""

    def __init__(self, param: HANQueryParamemter):
        """Initialize graph query."""
        self.param = param
        self.label_meta = np.array([[param.label_idx, param.label_dim]], np.int32)
        self.feat_meta = np.array([[param.feature_idx, param.feature_dim]], np.int32)
        assert param.nb_num != 0
        self.metapath_num = int(len(param.edge_types) / len(param.nb_num))

    def query_trainning(
        self, graph: Graph, inputs: np.ndarray, return_shape: bool = False
    ) -> tuple:
        """Generate data to train model."""
        if self.param.label_idx == -1:
            label = np.empty([len(inputs), self.param.label_dim], np.int32)
        else:
            label = graph.node_features(inputs, self.label_meta, FeatureType.INT64)

        hop_num = len(self.param.nb_num)

        total_nb_num = np.prod(self.param.nb_num)
        node_feats_arr = []
        neighbor_feats_arr = []
        for i in range(self.metapath_num):
            neighbors_list = multihop.sample_fanout(
                graph,
                inputs,
                self.param.edge_types[i * hop_num : (i + 1) * hop_num],
                self.param.nb_num,
                default_node=self.param.max_id + 1,
            )[0]

            neighbors = np.reshape(neighbors_list[-1], [-1, total_nb_num])
            node_feats = graph.node_features(inputs, self.feat_meta, FeatureType.FLOAT)
            neighbor_feats = graph.node_features(
                np.reshape(neighbors, [-1]), self.feat_meta, FeatureType.FLOAT
            )

            node_feats_arr.append(
                np.reshape(node_feats, [-1, 1, self.param.feature_dim])
            )
            neighbor_feats_arr.append(
                np.reshape(neighbor_feats, [-1, total_nb_num, self.param.feature_dim])
            )
        src_nodes = np.concatenate([inputs])
        nd_arr = np.array(node_feats_arr)
        nei_arr = np.array(neighbor_feats_arr)
        graph_tensor = (src_nodes, label, nd_arr, nei_arr)
        if return_shape:
            N = len(inputs)
            shapes = (
                [N],  # Nodes
                [N, self.param.label_dim],  # label
                [self.metapath_num, N, 1, self.param.feature_dim],  # feat
                [
                    self.metapath_num,
                    N,
                    total_nb_num,
                    self.param.feature_dim,
                ],  # neigobor feats
            )
            return graph_tensor, shapes

        return graph_tensor


class HAN(tf.keras.Model):
    """HAN model implementation."""

    def __init__(
        self,
        edge_types: list,
        nb_num: list,
        label_idx: int,
        label_dim: int,
        head_num: list,
        hidden_dim: list,
        feature_idx=-1,
        feature_dim=0,
        max_id=-1,
        loss_name="sigmoid",
    ):
        """Initialize HAN model."""
        super().__init__(name="han")
        assert loss_name in ["sigmoid", "softmax"]
        self.max_id = max_id
        self.loss_name = loss_name

        self.label_meta = np.array([[label_idx, label_dim]])
        self.encoder = encoders.HANEncoder(
            edge_types=edge_types,
            head_num=head_num,
            hidden_dim=hidden_dim,
            nb_num=nb_num,
            feature_idx=feature_idx,
            feature_dim=feature_dim,
            max_id=max_id,
        )

        self.num_classes = label_dim
        self.predict_layer = tf.keras.layers.Dense(self.num_classes, name="pred_layer")

    def embedding_fn(self):
        """
        Return output embeddings.

        * self.src_nodes: (batch_size, )
        * output_embedding: (batch_size, self.out_dim)
        """
        # self.src_embedding: (batch_size, 1, self.out_dim)
        output_embedding = tf.squeeze(self.src_emb)
        return [self.src_nodes, output_embedding]

    def call(
        self,
        inputs: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        training=True,
    ):
        """Generate embeddings, loss and F1 score."""
        src_nodes, labels, node_feats_arr, neighbor_feats_arr = inputs
        embedding = self.encoder((node_feats_arr, neighbor_feats_arr))
        if not training:
            self.src_emb = embedding
            self.src_nodes = src_nodes
            return embedding

        loss, f1 = self.decoder(embedding, labels)

        return embedding, loss, {"f1": f1}

    def f1_score(self, labels, predictions):
        """Compute F1 score from labels and predictions."""
        # TODO: remove this later. copied from deepgnn.tf.metrics
        with tf.compat.v1.variable_scope("f1", "f1", (labels, predictions)):
            epsilon = 1e-7
            _, tp = tf.compat.v1.metrics.true_positives(labels, predictions)
            _, fn = tf.compat.v1.metrics.false_negatives(labels, predictions)
            _, fp = tf.compat.v1.metrics.false_positives(labels, predictions)
            precision = tf.compat.v1.div(tp, epsilon + tp + fp, name="precision")
            recall = tf.compat.v1.div(tp, epsilon + tp + fn, name="recall")
            f1 = 2.0 * precision * recall / (precision + recall + epsilon)
        return f1

    def decoder(self, embeddings, labels):
        """Compute loss and F1 score."""
        logits = self.predict_layer(embeddings)
        if self.loss_name == "sigmoid":
            loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)
            predictions = tf.nn.sigmoid(logits)
            predictions = tf.floor(predictions + 0.5)
        elif self.loss_name == "softmax":
            loss = tf.nn.softmax_cross_entropy_with_logits(
                labels=tf.stop_gradient(labels), logits=logits
            )
            predictions = tf.nn.softmax(logits)
            predictions = tf.one_hot(
                tf.argmax(input=predictions, axis=1), self.num_classes
            )
        loss = tf.reduce_mean(input_tensor=loss)
        f1 = self.f1_score(labels, predictions)
        return loss, f1

    def train_step(self, data: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]):
        """Override base train_step."""
        with tf.GradientTape() as tape:
            _, loss, metrics = self(data, training=True)

        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        result = {"loss": loss}
        result.update(metrics)
        return result

    def test_step(self, data: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]):
        """Override base test_step."""
        _, loss, metrics = self(data, training=False)
        result = {"loss": loss}
        result.update(metrics)
        return result

    def predict_step(self, data: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]):
        """Override base predict_step."""
        self(data, training=False)
        return [self.src_nodes, self.src_emb]
