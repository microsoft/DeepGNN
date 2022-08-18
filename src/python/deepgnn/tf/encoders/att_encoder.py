# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Encoder for GAT model."""
import numpy as np
import tensorflow as tf

from deepgnn.graph_engine import Graph, FeatureType, QueryOutput
from deepgnn.tf import layers


class AttEncoder(layers.Layer):
    """Attention Encoder with neighbor sampling (https://arxiv.org/abs/1710.10903)."""

    def __init__(
        self,
        edge_type: int = 0,
        feature_idx: int = -1,
        feature_dim: int = 0,
        head_num: int = 1,
        hidden_dim: int = 256,
        nb_num: int = 5,
        out_dim: int = 1,
        **kwargs
    ):
        """Initialize encoder."""
        super(AttEncoder, self).__init__(**kwargs)
        self.features_metadata = np.array([[feature_idx, feature_dim]])
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.head_num = head_num
        self.nb_num = nb_num
        self.edge_type = np.array(edge_type, dtype=np.int32)
        self.hidden_header = [
            layers.AttentionHeader(self.hidden_dim, tf.nn.elu)
            for _ in range(self.head_num)
        ]
        self.out_header = [
            layers.AttentionHeader(self.out_dim, tf.nn.elu)
            for _ in range(self.head_num)
        ]

    def query(self, graph: Graph, context: QueryOutput) -> dict:
        """Fetch training data from graph."""
        neighbors = graph.sample_neighbors(
            context["inputs"], self.edge_type, self.nb_num
        )[0]
        node_feats = graph.node_features(
            context["inputs"], self.features_metadata, FeatureType.FLOAT
        )
        neighbor_feats = graph.node_features(
            np.reshape(neighbors, [-1]), self.features_metadata, FeatureType.FLOAT
        )
        context["node_feats"] = np.reshape(node_feats, [-1, 1, self.feature_dim])
        context["neighbor_feats"] = np.reshape(  # type: ignore
            neighbor_feats, [-1, self.nb_num, self.feature_dim]
        )
        return context

    def call(self, context: QueryOutput) -> tf.Tensor:
        """Compute embeddings."""
        seq = tf.concat(
            [context["node_feats"], context["neighbor_feats"]], 1
        )  # [bz,nb+1,fdim]

        hidden = []
        for i in range(0, self.head_num):
            hidden_val = self.hidden_header[i](seq)
            tf.compat.v1.logging.info("hidden shape {0}".format(hidden_val.shape))
            hidden_val = tf.reshape(hidden_val, [-1, self.nb_num + 1, self.hidden_dim])
            hidden.append(hidden_val)
        h_1 = tf.concat(hidden, -1)
        out = []
        for i in range(0, self.head_num):
            out_val = self.out_header[i](h_1)
            out_val = tf.reshape(out_val, [-1, self.nb_num + 1, self.out_dim])
            out.append(out_val)
        out = tf.add_n(out) / self.head_num
        out = tf.reshape(out, [-1, self.nb_num + 1, self.out_dim])
        out = tf.slice(out, [0, 0, 0], [-1, 1, self.out_dim])
        tf.compat.v1.logging.info("out shape {0}".format(out.shape))  # type: ignore
        return tf.reshape(out, [-1, self.out_dim])
