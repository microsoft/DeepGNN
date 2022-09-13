# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Reference: https://dl.acm.org/doi/pdf/10.1145/3219819.3219947."""
from typing import Callable, List
import torch
import torch.nn as nn
from deepgnn.pytorch.common.consts import (
    MODEL_RESIDUAL_ADD,
    MODEL_RESIDUAL_CONCAT,
    INPUTS,
    FANOUTS,
)


class LgclEncoder(nn.Module):
    """Encoder for learnable graph convolution nbetwork model."""

    OUTPUT = "lgcl_output"
    ADD_RES = "lgcl_add"
    LGCL_PREFIX = "lgcl"

    def __init__(
        self,
        in_features: int,
        largest_k_list: List[int],
        hidden_dims: List[int],
        residual: str = MODEL_RESIDUAL_ADD,
        acts: List[Callable[[torch.Tensor], torch.Tensor]] = [],
    ):
        """Initialize encoder."""
        super(LgclEncoder, self).__init__()

        self.largest_k_list = largest_k_list
        self.in_features = in_features
        self.residual = residual
        self.hidden_dims = hidden_dims
        self.acts = acts

        num_layers = len(largest_k_list)
        for layer in range(num_layers):
            for hop in range(num_layers - layer):
                k = largest_k_list[hop]
                self.add_module(
                    f"{LgclEncoder.LGCL_PREFIX}_{layer}_{hop}_0",
                    torch.nn.Conv1d(
                        in_features,
                        (in_features + self.hidden_dims[layer]) // 2,
                        (k + 1) // 2 + 1,
                    ),
                )
                self.add_module(
                    f"{LgclEncoder.LGCL_PREFIX}_{layer}_{hop}_1",
                    torch.nn.Conv1d(
                        (in_features + self.hidden_dims[layer]) // 2,
                        self.hidden_dims[layer],
                        k // 2 + 1,
                    ),
                )

            if self.residual == MODEL_RESIDUAL_ADD:
                if in_features != self.hidden_dims[layer]:
                    self.add_module(
                        f"{LgclEncoder.ADD_RES}_{str(layer)}",
                        nn.Linear(in_features, self.hidden_dims[layer]),
                    )
                in_features = self.hidden_dims[layer]
            elif self.residual == MODEL_RESIDUAL_CONCAT:
                in_features = self.hidden_dims[layer] + in_features

        if in_features != self.hidden_dims[-1]:
            self.add_module(
                f"{LgclEncoder.OUTPUT}", nn.Linear(in_features, self.hidden_dims[-1])
            )

    def forward(self, context: dict) -> torch.Tensor:
        """Evaluate encoder."""
        samples = context[INPUTS]
        fanouts = context[FANOUTS]

        num_layers = len(fanouts)
        for layer in range(num_layers):
            hidden = []
            for hop in range(num_layers - layer):
                neighbor = torch.reshape(
                    samples[hop + 1], [-1, fanouts[hop], samples[hop + 1].shape[-1]]
                )
                neighbor_topk = torch.topk(neighbor, self.largest_k_list[hop], dim=1)
                seq = torch.cat(
                    [torch.unsqueeze(samples[hop], 1), neighbor_topk.values], dim=1
                )
                seq = torch.transpose(seq, 1, 2)  # batch, feature, length
                seq = self.acts[layer](
                    getattr(self, f"{LgclEncoder.LGCL_PREFIX}_{layer}_{hop}_0")(seq)
                )
                seq = self.acts[layer](
                    getattr(self, f"{LgclEncoder.LGCL_PREFIX}_{layer}_{hop}_1")(seq)
                )
                seq = torch.squeeze(seq)

                if self.residual == MODEL_RESIDUAL_ADD:
                    if samples[hop].shape[-1] != seq.shape[-1]:
                        seq = seq + getattr(
                            self, f"{LgclEncoder.ADD_RES}_{str(layer)}"
                        )(samples[hop])
                    else:
                        seq = seq + samples[0]
                elif self.residual == MODEL_RESIDUAL_CONCAT:
                    seq = torch.cat([seq, samples[hop]], dim=-1)

                hidden.append(seq)
            samples = hidden

        if samples[0].shape[-1] != self.hidden_dims[-1]:
            ret = self.acts[-1](getattr(self, f"{LgclEncoder.OUTPUT}")(samples[0]))
        else:
            ret = samples[0]

        return ret
