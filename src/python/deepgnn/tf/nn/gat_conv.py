# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Layers for GAT model."""
import tensorflow as tf
from typing import Callable


def _sparse_dropout(x, rate):
    idx = x.indices
    val = x.values
    shp = x.dense_shape
    val = tf.nn.dropout(val, rate=rate)
    newx = tf.SparseTensor(idx, val, shp)
    return newx


class AttnHead(tf.keras.layers.Layer):
    """Attention Header: https://github.com/PetarV-/GAT/blob/master/utils/layers.py."""

    def __init__(
        self,
        out_dim: int,
        act: Callable = None,
        in_drop: float = 0.0,
        coef_drop: float = 0.0,
    ):
        """
        Attention Header Layer.

        Args:
          * out_dim: output feature dimension.
          * act: activation function.
          * in_drop: input feature dropout rate.
          * coef_drop: coeffeicent dropout rate.
        """
        super().__init__()
        self.out_dim = out_dim
        self.act = act
        self.in_drop = in_drop
        self.coef_drop = coef_drop

        self.w = tf.keras.layers.Dense(self.out_dim, use_bias=False, name="w")
        self.attn_l = tf.keras.layers.Dense(1, name="attn_l")
        self.attn_r = tf.keras.layers.Dense(1, name="attn_r")
        self.bias = self.add_weight(
            name="attn.bias",
            shape=[self.out_dim],
            initializer="zeros",
            dtype=tf.float32,
            trainable=True,
        )

    def call(self, inputs: tf.Tensor, training: bool = True) -> tf.Tensor:
        """Compute embeddings."""
        # feat: [N, F], adj: [N, N]
        feat, adj = inputs

        if self.in_drop != 0.0 and training:
            feat = tf.nn.dropout(feat, rate=self.in_drop)

        seq_fts = self.w(feat)  # [N, F']
        f_1 = self.attn_l(seq_fts)  # [N, 1]
        f_2 = self.attn_r(seq_fts)  # [N, 1]

        if self.in_drop != 0.0 and training:
            seq_fts = tf.nn.dropout(seq_fts, rate=self.in_drop)

        # fmt: off
        if isinstance(adj, tf.SparseTensor):
            vals, logits, coefs = self._call_sparse_version(seq_fts, f_1, f_2, adj, training)
        else:
            vals, logits, coefs = self._call_dense_version(seq_fts, f_1, f_2, adj, training)
        # fmt: on

        ret = tf.nn.bias_add(vals, self.bias)  # [N, F']

        tf.compat.v1.logging.info(
            f"Att Header: seq{feat.shape}, seq_flts {seq_fts.shape}, f_1 {f_1.shape}, f_2 {f_2.shape}, "
            f"logits {logits.shape}, coefs {coefs.shape}, vals {vals.shape} ret {ret.shape}"
        )
        if self.act is None:
            return ret
        else:
            return self.act(ret)

    def _call_dense_version(self, seq_fts, f_1, f_2, adj, training):
        bias_mat = -1e9 * (1.0 - adj)
        logits = f_1 + tf.transpose(a=f_2, perm=[1, 0])  # [N, N]-broadcasting
        coefs = tf.nn.softmax(tf.nn.leaky_relu(logits) + bias_mat)  # [N, N]

        if self.coef_drop != 0.0 and training:
            coefs = tf.nn.dropout(coefs, rate=self.coef_drop)

        vals = tf.matmul(coefs, seq_fts)  # [N, F']
        return vals, logits, coefs

    def _call_sparse_version(self, seq_fts, f_1, f_2, adj, training):
        edges = adj.indices  # [X, 2], X: num of edges
        adj_shape = adj.dense_shape  # [N, N]
        row = edges[:, 0]
        col = edges[:, 1]

        logits = tf.nn.embedding_lookup(f_1, row) + tf.nn.embedding_lookup(
            f_2, col
        )  # [X, 1]
        logits = tf.nn.leaky_relu(logits)
        coefs = tf.SparseTensor(edges, tf.reshape(logits, [-1]), adj_shape)
        coefs = tf.sparse.softmax(coefs)  # Sparse tensor (dense_shape: [N, N])

        if self.coef_drop != 0.0 and training:
            coefs = _sparse_dropout(coefs, self.coef_drop)

        vals = tf.sparse.sparse_dense_matmul(coefs, seq_fts)  # [N, F']
        return vals, logits, coefs


class GATConv(tf.keras.layers.Layer):
    """Graph Attention Conv Layer."""

    def __init__(
        self,
        out_dim: int,
        attn_heads: int,
        act: Callable = None,
        in_drop: float = 0.0,
        coef_drop: float = 0.0,
        attn_aggregate: str = "concat",
    ):
        """
        Initialize GAT convolution layer.

        Args:
          * out_dim: output feature dimension.
          * attn_heads: attention headers.
          * act: activation function.
          * in_drop: input feature dropout rate.
          * coef_drop: coeffeicent dropout rate.
          * attn_aggregate: concat or average.
        """
        if attn_aggregate not in {"concat", "average"}:
            raise ValueError(
                f"unknown attention aggregate function: {attn_aggregate}, only support aggregate methods: concat, average"
            )
        super().__init__()
        self.attn_heads = attn_heads
        self.attn_aggregate = attn_aggregate

        self.out_dim = out_dim
        self.act = act
        self.in_drop = in_drop
        self.coef_drop = coef_drop

        self.headers = [
            AttnHead(
                self.out_dim,
                act=tf.nn.elu,
                in_drop=self.in_drop,
                coef_drop=self.coef_drop,
            )
            for _ in range(self.attn_heads)
        ]

    def call(self, inputs: tf.Tensor, training: bool = True) -> tf.Tensor:
        """Compute embeddings."""
        attns = []
        for i in range(self.attn_heads):
            v = self.headers[i](inputs, training=training)
            attns.append(v)
        if self.attn_aggregate == "concat":
            h_1 = tf.concat(attns, -1)
        elif self.attn_aggregate == "average":
            h_1 = tf.add_n(attns) / self.attn_heads
        return h_1
