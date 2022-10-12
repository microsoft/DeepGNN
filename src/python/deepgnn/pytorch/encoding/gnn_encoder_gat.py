# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Encoder for GAT model."""
from typing import Optional, Callable
import torch
import torch.nn as nn
import torch.nn.functional as F


class GatEncoder(nn.Module):
    """Reference: https://arxiv.org/pdf/1710.10903.pdf."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        dropout: float = 0.2,
        negative_slope: float = 1e-2,
        concat: bool = True,
        act: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ):
        """Initialize encoder."""
        super(GatEncoder, self).__init__()

        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.negative_slope = negative_slope
        self.concat = concat
        self.act = act

        self.leakyrelu = nn.LeakyReLU(self.negative_slope)
        self.fc = nn.Linear(self.in_features, self.out_features, bias=False)
        self.attn_l = nn.Linear(self.out_features, 1)
        self.attn_r = nn.Linear(self.out_features, 1)

    def forward(self, combind_feats: torch.Tensor) -> torch.Tensor:
        """Evaluate encoder."""
        feats = self.fc(combind_feats)
        f_1 = self.attn_l(feats)
        f_2 = self.attn_r(feats)
        logits = f_1 + f_2.permute(0, 2, 1)
        coefs = F.softmax(self.leakyrelu(logits), dim=1)
        ret = torch.matmul(coefs, feats)

        if self.act is None:
            return ret
        else:
            return self.act(ret)
