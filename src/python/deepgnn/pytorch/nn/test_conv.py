# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import numpy as np
import random

from deepgnn.pytorch.nn.gat_conv import AttnHead


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def test_attn_head():
    N = 5
    D = 3

    set_seed(123)
    x = np.random.rand(N, D).astype(np.float32)  # [N, D]
    adj = np.random.randint(2, size=N * N).astype(np.float32).reshape(N, N)

    def run_dense_version():
        set_seed(123)
        attn_layer = AttnHead(D, 4, in_drop=0.2, coef_drop=0.0)
        x2 = torch.from_numpy(x)
        adj2 = torch.from_numpy(adj)
        out = attn_layer(x2, adj2)
        return out

    def run_sparse_version():
        set_seed(123)
        attn_layer = AttnHead(D, 4, in_drop=0.2, coef_drop=0.0)
        row, col = np.nonzero(adj)
        edge = np.concatenate([row.reshape(-1, 1), col.reshape(-1, 1)], axis=1)
        edge_value = np.ones(edge.shape[0], np.float32)
        edge = np.transpose(edge)
        sp_adj = torch.sparse_coo_tensor(edge, edge_value, [N, N])
        x2 = torch.from_numpy(x)
        out = attn_layer(x2, sp_adj)
        return out

    dense_out = run_dense_version()
    sparse_out = run_sparse_version()
    np.testing.assert_allclose(
        dense_out.detach().numpy(), sparse_out.detach().numpy(), atol=0.001
    )
