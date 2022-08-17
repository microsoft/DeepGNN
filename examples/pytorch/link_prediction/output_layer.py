# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Output layer implementation."""
import torch.nn as nn

from deepgnn.pytorch.encoding.twinbert.pooler import WeightPooler
from deepgnn.pytorch.encoding.twinbert.encoder import TwinBERTEncoder
from deepgnn.pytorch.encoding.twinbert.deepspeed.nvidia_modeling import (  # type: ignore
    BertEncoder,
    BertConfig,
)
from deepgnn.pytorch.encoding.twinbert.configuration import DeepSpeedArgs
from consts import (  # type: ignore
    SIM_TYPE_COSINE,
    SIM_TYPE_MAXFEEDFORWARD,
    SIM_TYPE_MAXRESLAYER,
    SIM_TYPE_RESLAYER,
    SIM_TYPE_FEEDFORWARD,
    SIM_TYPE_SELFATTENTION,
    HIDDEN_SIZE,
    SIM_TYPE_COSINE_WITH_RNS,
)
from poolers import (  # type: ignore
    feedforwardpooler,
    maxfeedforwardpooler,
    reslayerpooler,
    maxreslayerpooler,
    cosinepooler,
    cosinepooler_with_rns,
    selfattentionpooler,
)


class OutputLayer(nn.Module):
    """
    OutputLayer is a crossing layer.

    It is used to calculate the similarity of the source node and destination node based on different poolers::
        cosine,
        cosine_with_rns,
        feedforward,
        maxfeedforward,
        reslayer,
        maxreslayer,
        selfattentionю
    """

    def __init__(
        self,
        input_dim,
        sim_type: str = SIM_TYPE_COSINE,
        res_size: int = 64,
        res_bn: bool = False,
        featenc_config: str = "",
        random_negative: int = 0,
        nsp_gamma: int = 1,
    ):
        """Initialize output layer."""
        super(OutputLayer, self).__init__()
        self.crossing_layers = []  # type: ignore
        self.sim_type = sim_type
        self.res_size = res_size
        self.res_bn = res_bn
        self.featenc_config = featenc_config
        self.nsp_gamma = nsp_gamma
        self.random_negative = random_negative

        hidden_size = input_dim

        if self.sim_type not in [SIM_TYPE_MAXFEEDFORWARD, SIM_TYPE_MAXRESLAYER]:
            hidden_size = hidden_size * 2

        if self.sim_type == SIM_TYPE_COSINE:
            self.crossing_layers = [nn.Linear(1, 1)]

        if SIM_TYPE_RESLAYER in self.sim_type:
            self.crossing_layers = [
                nn.Linear(hidden_size, self.res_size),
                nn.Linear(self.res_size, hidden_size),
                nn.Linear(hidden_size, 1),
                nn.BatchNorm1d(self.res_size) if self.res_bn else None,  # type: ignore
            ]

        if SIM_TYPE_FEEDFORWARD in self.sim_type:
            self.crossing_layers = [
                nn.Linear(hidden_size, hidden_size),
                nn.Linear(hidden_size, 1),
            ]

        if self.sim_type == SIM_TYPE_SELFATTENTION:
            config = TwinBERTEncoder.init_config_from_file(self.featenc_config)
            self.crossing_layers = [
                BertEncoder(BertConfig.from_dict(config), DeepSpeedArgs(config)),
                WeightPooler(config),
                nn.Linear(config[HIDDEN_SIZE], 1),
            ]

        for i in range(len(self.crossing_layers)):
            self.add_module(f"crossing_layer{i}", self.crossing_layers[i])

    def simpooler(
        self,
        src_vec,
        src_term_tensor,
        src_mask,
        dst_vec,
        dst_term_tensor,
        dst_mask,
        with_rng=False,
        prefix="",
    ):
        """Calculate similarity between src and dst vectors."""
        results = None
        if self.sim_type == SIM_TYPE_COSINE:
            results = cosinepooler(
                src_vec,
                dst_vec,
                with_rng,
                self.crossing_layers,
                random_negative=self.random_negative,
            )
        elif self.sim_type == SIM_TYPE_COSINE_WITH_RNS:
            results = cosinepooler_with_rns(
                src_vec,
                dst_vec,
                with_rng,
                nsp_gamma=self.nsp_gamma,
                random_negative=self.random_negative,
            )
        elif self.sim_type == SIM_TYPE_RESLAYER:
            results = reslayerpooler(
                src_vec,
                dst_vec,
                with_rng,
                self.crossing_layers,
                random_negative=self.random_negative,
            )
        elif self.sim_type == SIM_TYPE_MAXRESLAYER:
            results = maxreslayerpooler(
                src_vec,
                dst_vec,
                with_rng,
                self.crossing_layers,
                random_negative=self.random_negative,
            )
        elif self.sim_type == SIM_TYPE_FEEDFORWARD:
            results = feedforwardpooler(
                src_vec,
                dst_vec,
                with_rng,
                self.crossing_layers,
                random_negative=self.random_negative,
            )
        elif self.sim_type == SIM_TYPE_MAXFEEDFORWARD:
            results = maxfeedforwardpooler(
                src_vec,
                dst_vec,
                with_rng,
                self.crossing_layers,
                random_negative=self.random_negative,
            )
        elif self.sim_type == SIM_TYPE_SELFATTENTION:
            results = selfattentionpooler(
                src_vec,
                src_term_tensor,
                src_mask,
                dst_vec,
                dst_term_tensor,
                dst_mask,
                self.crossing_layers,
                self.random_negative,
                with_rng,
            )

        return results
