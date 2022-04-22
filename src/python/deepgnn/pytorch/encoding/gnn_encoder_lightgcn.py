# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import torch.nn as nn
from deepgnn.pytorch.common.consts import INPUTS, FANOUTS


class LightGCNEncoder(nn.Module):
    def __init__(self):
        super(LightGCNEncoder, self).__init__()

    def forward(self, context: dict):
        samples = context[INPUTS]
        fanouts = context[FANOUTS]

        num_layers = len(fanouts)
        if num_layers == 0:
            return samples[0]

        assert num_layers == 1
        neighbor = torch.reshape(samples[1], [-1, fanouts[0], samples[1].shape[-1]])
        neighbor = torch.mean(neighbor, dim=1)
        seq = (samples[0] + neighbor) / 2
        return seq
