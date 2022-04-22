# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import tensorflow as tf
import numpy as np

from deepgnn.tf.nn.gcn_conv import gcn_norm_adj
from deepgnn.tf.nn.gat_conv import AttnHead


def set_seed(seed):
    tf.random.set_seed(seed)
    np.random.seed(seed)


def test_attn_head():
    N = 5
    D = 3

    set_seed(123)
    x = np.random.rand(N, D).astype(np.float32)  # [N, D]
    adj = np.random.randint(2, size=N * N).astype(np.float32).reshape(N, N)

    def run_dense_version():
        set_seed(123)
        attn_layer = AttnHead(4, act=tf.nn.elu, in_drop=0.1, coef_drop=0.0)
        out = attn_layer([x, adj])
        return out

    def run_sparse_version():
        set_seed(123)
        attn_layer = AttnHead(4, act=tf.nn.elu, in_drop=0.1, coef_drop=0.0)
        row, col = np.nonzero(adj)
        edge = np.concatenate([row.reshape(-1, 1), col.reshape(-1, 1)], axis=1)
        edge_value = np.ones(edge.shape[0], np.float32)
        adj_shape = np.array([N, N], np.int64)
        sp_adj = tf.SparseTensor(edge, edge_value, adj_shape)
        attn_layer = AttnHead(4, act=tf.nn.elu, in_drop=0.1, coef_drop=0.0)
        out = attn_layer([x, sp_adj])
        return out

    # fmt: off
    expected_out = np.array(
        [[ 0.13966814, -0.43618077, 1.1495067, 0.35430294],
         [ 0.10271902, -0.52295756, 0.5240147, 0.25755188],
         [-0.0497849, -0.49666244, 0.68464315, 0.2244271],
         [ 0.4595074, -0.5716846, 0.7725301, 0.],
         [ 0.22125529, -0.539732, 0.6065793, 0.1719851]],
        np.float32,
    )
    # fmt: on
    dense_out = run_dense_version()
    tf.debugging.assert_near(dense_out, expected_out, atol=0.0001)

    _ = run_sparse_version()
    tf.debugging.assert_near(dense_out, expected_out, atol=0.0001)


def test_gcn_norm():
    N = 5
    D = 3
    set_seed(123)
    _ = np.random.rand(N, D)  # [N, D]
    adj = np.random.randint(2, size=N * N).astype(np.float32).reshape(N, N)

    def run_dense_adj(raw_adj):
        adj1 = gcn_norm_adj(raw_adj)
        return adj1

    def run_sparse_adj(raw_adj):
        row, col = np.nonzero(raw_adj)
        edge = np.concatenate([row.reshape(-1, 1), col.reshape(-1, 1)], axis=1)
        edge_value = np.ones(edge.shape[0], np.float32)
        adj_shape = np.array([N, N], np.int64)
        sp_adj_raw = tf.sparse.SparseTensor(edge, edge_value, adj_shape)
        sp_adj = gcn_norm_adj(sp_adj_raw)
        return sp_adj

    norm_adj1 = run_dense_adj(adj)
    assert isinstance(norm_adj1, tf.Tensor)

    norm_adj2_sp = run_sparse_adj(adj)
    norm_adj2 = tf.sparse.to_dense(norm_adj2_sp)
    assert isinstance(norm_adj2_sp, tf.SparseTensor)
    tf.debugging.assert_equal(norm_adj1, norm_adj2)
