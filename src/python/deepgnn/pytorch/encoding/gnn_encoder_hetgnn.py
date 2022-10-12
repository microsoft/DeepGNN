# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Encoder for HetGNN model."""
from typing import Optional, Callable, List
import torch
import torch.nn as nn
from deepgnn.pytorch.common.consts import INPUTS, FANOUTS


class HetGnnEncoder(nn.Module):
    """
    HetGNN encoder.

    Reference: https://dl.acm.org/doi/pdf/10.1145/3292500.3330961
    """

    CUSTOM_LAYER = "hetGNN_custom_layer"
    BILSTM_LAYER = "hetGNN_bilstm"
    OUTPUT_LAYER = "hetGNN_out"
    ATT_LAYER = "hetGNN_attention"
    NODE_LAYER = "hetGNN_node"

    def __init__(
        self,
        in_features: int,
        hidden_dims: List[int],
        fanouts: list,
        encoders: Optional[List[nn.Module]] = None,
        acts: List[Callable[[torch.Tensor], torch.Tensor]] = [],
    ):
        """Initialize encoder."""
        super(HetGnnEncoder, self).__init__()

        self.hidden_dims = hidden_dims
        self.in_features = in_features
        self.fanouts = fanouts
        self.encoders = encoders
        self.acts = acts

        outDim = self.hidden_dims[-1]
        if len(fanouts) == 0 and in_features != outDim:
            self.add_module(
                f"{HetGnnEncoder.OUTPUT_LAYER}", nn.Linear(in_features, outDim)
            )
        else:
            for i, fanout in enumerate(fanouts):
                if self.encoders is not None and len(self.encoders) > 0:
                    self.add_module(
                        f"{HetGnnEncoder.CUSTOM_LAYER}_{str(i)}", self.encoders[i]
                    )
                else:  # bi-lstm, support one hop
                    self.add_module(
                        f"{HetGnnEncoder.BILSTM_LAYER}_{str(i)}",
                        nn.LSTM(in_features, int(outDim / 2), 1, bidirectional=True),
                    )

            if len(fanouts) > 1:
                if in_features != outDim:
                    self.add_module(
                        f"{HetGnnEncoder.NODE_LAYER}", nn.Linear(in_features, outDim)
                    )
                self.add_module(f"{HetGnnEncoder.ATT_LAYER}", nn.Linear(outDim * 2, 1))

    def forward(self, context: dict) -> torch.Tensor:
        """Evaluate encoder."""
        samples = context[INPUTS]

        if len(self.fanouts) == 0:
            if samples[0][0].shape[-1] != self.hidden_dims[-1]:
                return self.acts[-1](
                    getattr(self, f"{HetGnnEncoder.OUTPUT_LAYER}")(samples[0][0])
                )
            else:
                return samples[0][0]

        embed_list = []
        for i, fanout in enumerate(self.fanouts):
            if self.encoders is not None and len(self.encoders) > 0:
                ctx = {INPUTS: samples[0] + samples[i + 1], FANOUTS: fanout}
                embed_list.append(
                    getattr(self, f"{HetGnnEncoder.CUSTOM_LAYER}_{str(i)}")(ctx)
                )
            else:  # bi-lstm, only one hop
                neighbor = torch.reshape(
                    samples[i + 1][0], [-1, fanout[0], samples[i + 1][0].shape[-1]]
                )
                seq = torch.cat([torch.unsqueeze(samples[0][0], 1), neighbor], dim=1)
                seq = torch.transpose(seq, 0, 1)
                all_state, _ = getattr(self, f"{HetGnnEncoder.BILSTM_LAYER}_{str(i)}")(
                    seq
                )
                embed_list.append(torch.mean(all_state, 0))

        if len(self.fanouts) == 1:
            return embed_list[0]

        node = samples[0][0]
        if node.shape[-1] != self.hidden_dims[-1]:
            node = self.acts[-1](getattr(self, f"{HetGnnEncoder.NODE_LAYER}")(node))

        embed_list.append(node)
        embed_list2 = [torch.cat([node, emb], dim=1) for emb in embed_list]
        concate_embed = torch.stack(embed_list2, dim=1)
        atten_w = nn.functional.leaky_relu(
            getattr(self, f"{HetGnnEncoder.ATT_LAYER}")(concate_embed)
        )
        atten_w = nn.functional.softmax(atten_w, dim=1)
        atten_w = torch.reshape(atten_w, [-1, 1, len(embed_list)])
        output = torch.matmul(atten_w, torch.stack(embed_list, dim=1))

        return torch.reshape(output, [-1, self.hidden_dims[-1]])
