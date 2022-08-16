# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Modules for GAT models."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, Optional


class AttnHead(nn.Module):
    """Attention Header: https://github.com/PetarV-/GAT/blob/master/utils/layers.py."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        act: Optional[Callable[[torch.Tensor], torch.Tensor]] = F.elu,
        in_drop: float = 0.0,
        coef_drop: float = 0.0,
    ):
        """
        Initialize Attention Header.

        Args:
          * in_dim: input feature dimension
          * out_dim: output feature dimension.
          * act: activation function.
          * in_drop: input feature dropout rate.
          * coef_drop: coeffeicent dropout rate.
        """
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.act = act
        self.in_drop = torch.nn.Dropout(in_drop)
        self.coef_drop = torch.nn.Dropout(coef_drop)

        self.w = nn.Linear(self.in_dim, self.out_dim, bias=False)
        self.attn_l = nn.Linear(self.out_dim, 1)
        self.attn_r = nn.Linear(self.out_dim, 1)
        # TODO: add bias
        self.bias = nn.Parameter(torch.zeros(self.out_dim))

    def forward(self, feat: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """Evaluate module."""
        # feat: [N, F], adj: [N, N]
        if self.training:
            feat = self.in_drop(feat)

        seq_fts = self.w(feat)  # [N, F']
        f_1 = self.attn_l(seq_fts)  # [N, 1]
        f_2 = self.attn_r(seq_fts)  # [N, 1]

        if adj.is_sparse:
            vals = self._call_sparse_version(seq_fts, f_1, f_2, adj)
        else:
            vals = self._call_dense_version(seq_fts, f_1, f_2, adj)

        ret = vals + self.bias  # [N, F'], broadcast bias
        if self.act is None:
            return ret
        else:
            return self.act(ret)

    def _call_dense_version(
        self,
        seq_fts: torch.Tensor,
        f_1: torch.Tensor,
        f_2: torch.Tensor,
        adj: torch.Tensor,
    ) -> torch.Tensor:
        bias_mat = -1e9 * (1.0 - adj)
        logits = f_1 + f_2.transpose(1, 0)  # [N, N]-broadcasting
        assert logits.shape[0] == logits.shape[1]
        coefs = F.softmax(F.leaky_relu(logits, 0.2) + bias_mat, dim=1)  # [N, N]

        if self.training:
            coefs = self.coef_drop(coefs)
            seq_fts = self.in_drop(seq_fts)

        vals = torch.matmul(coefs, seq_fts)  # [N, F']
        return vals

    def _call_sparse_version(
        self,
        seq_fts: torch.Tensor,
        f_1: torch.Tensor,
        f_2: torch.Tensor,
        adj: torch.Tensor,
    ) -> torch.Tensor:
        indices = adj._indices()
        adj_shape = adj.shape
        row = indices[0]
        col = indices[1]
        logits = f_1[row] + f_2[col]
        logits = F.leaky_relu(logits, 0.2)
        coefs = torch.sparse_coo_tensor(indices, torch.reshape(logits, [-1]), adj_shape)
        coefs = torch.sparse.softmax(coefs, 1)

        if self.training:
            # TODO: current nn.Dropout() doesn't support sparse tesnor.
            # coefs = self.coef_drop(coefs)
            seq_fts = self.in_drop(seq_fts)

        vals = torch.sparse.mm(coefs, seq_fts)  # [N, F']

        return vals


class GATConv(nn.Module):
    """Graph Attention Convolution Layer."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        attn_heads: int,
        act: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        in_drop: float = 0.0,
        coef_drop: float = 0.0,
        attn_aggregate: str = "concat",
    ):
        """
        Initialize GAT convolution layer.

        Args:
          * in_dim: input feature dimension
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
                in_dim,
                self.out_dim,
                act=F.elu,
                in_drop=self.in_drop,
                coef_drop=self.coef_drop,
            )
            for _ in range(self.attn_heads)
        ]
        for i in range(self.attn_heads):
            self.add_module(f"att_head-{i}", self.headers[i])

    def forward(self, feat: torch.Tensor, bias_mat: torch.Tensor) -> torch.Tensor:
        """Evaluate module."""
        attns = []
        for i in range(self.attn_heads):
            v = self.headers[i](feat, bias_mat)
            attns.append(v)
        if self.attn_aggregate == "concat":
            h_1 = torch.cat(attns, -1)
        elif self.attn_aggregate == "average":
            # attns: [[N, F'], [N, F']...]
            h_1 = torch.stack(attns)  # [attn_heads, N, F']
            h_1 = h_1.sum(dim=0) / self.attn_heads  # [N, F']
        return h_1
