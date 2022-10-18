# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""GCN model implementation."""

import numpy as np
import tensorflow as tf
from dataclasses import dataclass

from deepgnn.tf.nn.gcn_conv import GCNConv, gcn_norm_adj
from deepgnn.tf.nn.metrics import masked_accuracy, masked_softmax_cross_entropy

from deepgnn.graph_engine import Graph, FeatureType, graph_ops

from deepgnn import get_logger


@dataclass
class GCNQueryParameter:
    """Graph query configuration."""

    neighbor_edge_types: np.ndarray
    feature_idx: int
    feature_dim: int
    label_idx: int
    label_dim: int
    feature_type: FeatureType = FeatureType.FLOAT
    label_type: FeatureType = FeatureType.FLOAT
    num_hops: int = 2


class GCNQuery:
    """Graph Query: get sub graph for GCN training."""

    def __init__(self, param: GCNQueryParameter):
        """Initialize GCNQuery."""
        self.param = param
        self.label_meta = np.array([[param.label_idx, param.label_dim]], np.int32)
        self.feat_meta = np.array([[param.feature_idx, param.feature_dim]], np.int32)

    def query_training(
        self, graph: Graph, inputs: np.ndarray, return_shape: bool = False
    ) -> tuple:
        """Query function to train a GCN model."""
        nodes, edges, src_idx = graph_ops.sub_graph(
            graph=graph,
            src_nodes=inputs,
            edge_types=self.param.neighbor_edge_types,
            num_hops=self.param.num_hops,
            self_loop=True,
            undirected=True,
            return_edges=True,
        )
        input_mask = np.zeros(nodes.size, np.bool)
        input_mask[src_idx] = True

        feat = graph.node_features(nodes, self.feat_meta, self.param.feature_type)
        label = graph.node_features(nodes, self.label_meta, self.param.label_type)
        label = label.astype(np.int32)

        edges_value = np.ones(edges.shape[0], np.float32)
        adj_shape = np.array([nodes.size, nodes.size], np.int64)
        graph_tensor = (nodes, feat, input_mask, label, edges, edges_value, adj_shape)
        if return_shape:
            # fmt: off
            # N is the number of `nodes`, which is variable because `inputs` nodes are different.
            N = None
            shapes = (
                [N],                            # Nodes
                [N, self.param.feature_dim],    # feat
                [N],                            # input_mask
                [N, self.param.label_dim],      # label
                [None, 2],                      # edges
                [None],                         # edges_value
                [2]                             # adj_shape
            )
            # fmt: on
            return graph_tensor, shapes

        return graph_tensor


class GCN(tf.keras.Model):
    """GCN Model (supervised)."""

    def __init__(
        self,
        hidden_dim: int = 8,
        num_classes: int = -1,
        dropout: float = 0.0,
        l2_coef: float = 0.0005,
    ):
        """Initialize GCN model."""
        super().__init__()
        self.num_classes = num_classes
        self.l2_coef = l2_coef
        self.out_dim = num_classes

        # fmt: off
        self.gcn_layers = [
            GCNConv(out_dim=hidden_dim, dropout=dropout, act=tf.nn.relu, use_bias=False),
            GCNConv(out_dim=self.out_dim, dropout=dropout, act=tf.nn.relu, use_bias=False),
        ]
        self.logger = get_logger()
        # fmt: on

    def forward(self, feat, adj, training):
        """Generate embeddings."""
        activations = [feat]
        for i, layer in enumerate(self.gcn_layers):
            hidden = layer([activations[-1], adj], training=training)
            self.logger.info(f"hidden layer {i} {hidden.shape}")
            activations.append(hidden)

        output = activations[-1]
        self.logger.info(f"output layer {i} {output.shape}")
        return output

    def call(self, inputs, training=True):
        """Calculate embeddings, loss and accuracy."""
        # inputs: nodes    feat      mask    labels   edges       edges_value  adj_shape
        # shape:  [N]      [N, F]    [N]     [N]      [num_e, 2]  [num_e]      [2]
        nodes, feat, mask, labels, edges, edges_value, adj_shape = inputs
        adj = tf.sparse.SparseTensor(edges, edges_value, adj_shape)
        adj = gcn_norm_adj(adj)
        logits = self.forward(feat, adj, training)

        # embedding results
        self.src_emb = tf.boolean_mask(logits, mask)
        self.src_nodes = tf.boolean_mask(nodes, mask)

        labels = tf.one_hot(labels, self.num_classes)
        logits = tf.reshape(logits, [-1, self.num_classes])
        labels = tf.reshape(labels, [-1, self.num_classes])
        mask = tf.reshape(mask, [-1])

        # loss
        xent_loss = masked_softmax_cross_entropy(logits, labels, mask)
        loss = xent_loss + self.l2_loss()

        # metric
        acc = masked_accuracy(logits, labels, mask)
        return logits, loss, {"accuracy": acc}

    def l2_loss(self):
        """Calculate l2 loss."""
        vs = []
        for v in self.trainable_variables:
            vs.append(tf.nn.l2_loss(v))
        lossL2 = tf.add_n(vs) * self.l2_coef
        return lossL2

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
