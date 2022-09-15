# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Encoders for link prediction models."""
import torch
import torch.nn as nn
import numpy as np

from deepgnn.pytorch.common.consts import (
    FANOUTS,
    INPUTS,
    EMBS_LIST,
    TERM_TENSOR,
    ENCODER_MASK,
)
from deepgnn.pytorch.encoding import (
    GatEncoder,
    LgclEncoder,
    HetGnnEncoder,
    LightGCNEncoder,
)
from consts import (  # type: ignore
    FANOUTS_NAME,
    ENCODER_LABEL,
    ENCODER_GAT,
    ACT_FUN_ELU,
    ACT_FUN_RELU,
    ACT_FUN_LEAKY_RELU,
    ACT_FUN_TANH,
    MODEL_GAT,
    MODEL_INIT,
    ENCODER,
    MODEL_LGCL,
    MODEL_LIGHTGCN,
    MODEL_HETGNN,
)


class GnnEncoder(nn.Module):
    """Generic encoder implementations for GNNs."""

    def __init__(
        self,
        input_dim,
        fanouts_dict=None,
        act_functions="",
        encoder_name="",
        head_nums=None,
        hidden_dims=None,
        lgcl_largest_k=0,
        residual="",
    ):
        """Initialize encoder."""
        super(GnnEncoder, self).__init__()

        self.head_nums = head_nums
        self.hidden_dims = hidden_dims
        self.lgcl_largest_k = lgcl_largest_k
        self.residual = residual

        self.gnn_activations = []
        for act in act_functions.split(","):
            if act == ACT_FUN_ELU:
                activation = nn.functional.elu
            elif act == ACT_FUN_RELU:
                activation = nn.functional.relu
            elif act == ACT_FUN_LEAKY_RELU:
                activation = nn.functional.leaky_relu
            elif act == ACT_FUN_TANH:
                activation = torch.tanh
            else:
                activation = None
            self.gnn_activations.append(activation)

        self.gnn_encoder_map = {
            MODEL_GAT: {MODEL_INIT: self.gat_init, ENCODER: self.gat_encoder},
            MODEL_LGCL: {MODEL_INIT: self.lgcl_init, ENCODER: self.lgcl_encoder},
            MODEL_HETGNN: {MODEL_INIT: self.hetGNN_init, ENCODER: self.hetGNN_encoder},
            MODEL_LIGHTGCN: {
                MODEL_INIT: self.light_gcn_init,
                ENCODER: self.light_gcn_encoder,
            },
        }

        self.init_and_encoder = []
        enc_name_list = encoder_name.lower().split("_")
        for i, enc_name in enumerate(enc_name_list):
            if enc_name in self.gnn_encoder_map:
                self.init_and_encoder.append(self.gnn_encoder_map[enc_name])
            else:
                raise ValueError(f"Unsupported GNN encoder name: {enc_name}.")

        assert len(self.init_and_encoder) > 0
        for key in fanouts_dict:
            if len(fanouts_dict[key]) > 0:
                fanouts = (
                    fanouts_dict[key]
                    if enc_name_list[0] == MODEL_HETGNN
                    else fanouts_dict[key][0]
                    if len(fanouts_dict[key]) == 1
                    else [np.sum(fanouts_dict[key])]
                )
                encoder_map = self.init_and_encoder[0][MODEL_INIT](
                    input_dim, fanouts, key
                )
                for enc_key in encoder_map:
                    self.add_module(enc_key, encoder_map[enc_key])

    def gat_init(self, gnn_input_dim, fanouts, prefix=""):
        """Initialize GAT encoder."""
        encoder_map = {}
        for layer, head_num in enumerate(self.head_nums):
            gnn_outout_dim = self.hidden_dims[layer]
            for head in range(head_num):
                # reuse the current GAT encoder.
                gat_enc = GatEncoder(
                    in_features=gnn_input_dim,
                    out_features=gnn_outout_dim,
                    act=self.gnn_activations[layer],
                )
                encoder_map[f"{prefix + ENCODER_GAT}_{layer}_{head}"] = gat_enc

            gnn_input_dim = gnn_outout_dim * head_num

        return encoder_map

    def gat_encoder(self, samples, fanouts, prefix=""):
        """Evaluate GAT encoder."""
        if len(fanouts) == 0:
            return samples[0]

        neighbor = torch.reshape(samples[1], [-1, fanouts[0], samples[0].shape[-1]])
        seq = torch.cat([torch.unsqueeze(samples[0], 1), neighbor], dim=1)

        for layer, head_num in enumerate(self.head_nums):
            hidden = []
            for i in range(head_num):
                hidden_val = getattr(self, f"{prefix + ENCODER_GAT}_{layer}_{i}")(seq)
                hidden.append(hidden_val)
            seq = torch.cat(hidden, -1)

        att_output = torch.stack(hidden, 1)
        att_output = att_output[:, :, 0, :]
        att_output = torch.mean(att_output, 1)
        return att_output

    def light_gcn_init(self, gnn_input_dim, fanouts, prefix=""):
        """Initialize light GCN encoder."""
        return {f"{prefix}_lightgcn_encoder": LightGCNEncoder()}

    def light_gcn_encoder(self, samples, fanouts, prefix=""):
        """Evaluate light GCN encoder."""
        context = {INPUTS: samples, FANOUTS: fanouts}
        return getattr(self, f"{prefix}_lightgcn_encoder")(context)

    def hetGNN_init(self, gnn_input_dim, fanouts, prefix=""):
        """Initialize light hetGNN encoder."""
        encoders = []

        # if het gnn needs other encoder such as lgcl. e.g. hetgnn_lgcl/hetgnn_gat
        if len(self.init_and_encoder) > 1:
            for i, fanout in enumerate(fanouts):
                enc_map = self.init_and_encoder[1][MODEL_INIT](
                    gnn_input_dim, fanout, prefix
                )
                for key in enc_map:
                    encoders.append(enc_map[key])

        return {
            f"{prefix}_hetgnn_encoder": HetGnnEncoder(
                in_features=gnn_input_dim,
                hidden_dims=self.hidden_dims,
                fanouts=fanouts,
                encoders=encoders,
                acts=self.gnn_activations,
            )
        }

    def hetGNN_encoder(self, samples, fanouts, prefix=""):
        """Evaluate light GCN encoder."""
        context = {INPUTS: samples}

        return getattr(self, f"{prefix}_hetgnn_encoder")(context)

    def lgcl_init(self, gnn_input_dim, fanouts, prefix=""):
        """Initialize LGCL encoder."""
        return {
            f"{prefix}_lgcl_encoder": LgclEncoder(
                in_features=gnn_input_dim,
                largest_k_list=[
                    x if self.lgcl_largest_k == 0 else self.lgcl_largest_k
                    for x in fanouts
                ],
                hidden_dims=self.hidden_dims,
                residual=self.residual,
                acts=self.gnn_activations,
            )
        }

    def lgcl_encoder(self, samples, fanouts, prefix=""):
        """
        Learnable graph convolution encoder.

        Reference: https://dl.acm.org/doi/pdf/10.1145/3219819.3219947.
        """
        context = {INPUTS: samples, FANOUTS: fanouts}

        return getattr(self, f"{prefix}_lgcl_encoder")(context)

    def forward(self, context):
        """Evaluate saved encoder and return term tensor with mask and label."""
        embs_list = context[EMBS_LIST]
        term_tensors = context[TERM_TENSOR]
        mask = context[ENCODER_MASK]
        label = context[ENCODER_LABEL]
        fanouts = context[FANOUTS]
        fanouts_name = context[FANOUTS_NAME]

        assert len(fanouts) > 0

        if self.init_and_encoder[0][ENCODER] == self.hetGNN_encoder:
            vec_gnn = self.init_and_encoder[0][ENCODER](
                embs_list, fanouts, fanouts_name
            )
        else:
            neighbors = (
                embs_list[1]
                if len(fanouts) == 1
                else self.merge_neighbors(embs_list[1:], fanouts)
            )
            vec_gnn = self.init_and_encoder[0][ENCODER](
                embs_list[0] + neighbors, [np.sum(fanouts)], fanouts_name
            )

        return vec_gnn, term_tensors, mask, label

    def merge_neighbors(self, embeds, fanouts):
        """Merge multi-type one-hop neighbors."""
        merges = []
        dim = embeds[0][0].shape[-1]
        for i, fanout in enumerate(fanouts):
            for j in fanout:
                merges.append(torch.reshape(embeds[i][0], [-1, j, dim]))
        merge = torch.cat(merges, dim=1)
        return [torch.reshape(merge, [-1, dim])]
