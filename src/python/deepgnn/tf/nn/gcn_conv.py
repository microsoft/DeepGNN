# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Convolution functions for GCN model."""
import tensorflow as tf
from typing import Callable


def gcn_norm_adj(adj: tf.Tensor) -> tf.Tensor:
    """Symmetrically normalize adjacency matrix."""
    if isinstance(adj, tf.SparseTensor):
        edges = adj.indices
        edges_value = adj.values
        adj_shape = adj.dense_shape
        degree = tf.scatter_nd(
            tf.reshape(edges[:, 0], (-1, 1)), edges_value, [adj_shape[0]]
        )
        d_inv = tf.math.pow(degree, -0.5)
        d_inv = tf.where(tf.math.is_inf(d_inv), tf.zeros_like(d_inv), d_inv)
        d_inv_src = tf.nn.embedding_lookup(d_inv, edges[:, 0])
        d_inv_dst = tf.nn.embedding_lookup(d_inv, edges[:, 1])
        edge_weight = d_inv_src * d_inv_dst
        sp_adj = tf.sparse.SparseTensor(edges, edge_weight, dense_shape=adj_shape)
        return sp_adj
    else:
        rowsum = tf.reduce_sum(adj, axis=1)
        d_inv_sqrt = tf.math.pow(rowsum, -0.5)
        d_inv_sqrt = tf.where(
            tf.math.is_inf(d_inv_sqrt), tf.zeros_like(d_inv_sqrt), d_inv_sqrt
        )
        d_mat_inv_sqrt = tf.linalg.diag(d_inv_sqrt)
        return d_mat_inv_sqrt @ adj @ d_mat_inv_sqrt


class GCNConv(tf.keras.layers.Layer):
    """Graph Conv Layer."""

    def __init__(
        self,
        out_dim: int,
        dropout: float = 0.0,
        act: Callable = tf.nn.relu,
        use_bias: bool = False,
    ):
        """Initialize convolution layer."""
        # TODO: support sparse input
        super().__init__()

        self.out_dim = out_dim
        self.act = act
        self.dropout = dropout
        self.use_bias = use_bias

        self.w = tf.keras.layers.Dense(self.out_dim, use_bias=False, name="w")
        self.bias = self.add_weight(
            name="gcn.bias",
            shape=[self.out_dim],
            initializer="zeros",
            dtype=tf.float32,
            trainable=True,
        )

    def call(self, inputs: tf.Tensor, training: bool = True) -> tf.Tensor:
        """Compute embeddings."""
        x, adj = inputs
        x = tf.nn.dropout(x, rate=self.dropout)
        x = self.w(x)
        if isinstance(adj, tf.SparseTensor):
            support = tf.sparse.sparse_dense_matmul(adj, x)
        else:
            support = tf.matmul(adj, x, a_is_sparse=True)
        # output = tf.add_n(supports) # skip this because len(support) == 1
        output = support

        if self.use_bias:
            output = tf.nn.bias_add(output, self.bias)  # [N, F']
        return output
