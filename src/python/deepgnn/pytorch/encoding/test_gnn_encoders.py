# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import numpy as np
import random
import torch
import numpy.testing as npt
import torch.nn as nn

from deepgnn.pytorch.common.consts import INPUTS, FANOUTS
from deepgnn.pytorch.encoding.gnn_encoder_lightgcn import LightGCNEncoder
from deepgnn.pytorch.encoding.gnn_encoder_gat import GatEncoder
from deepgnn.pytorch.encoding.gnn_encoder_hetgnn import HetGnnEncoder
from deepgnn.pytorch.encoding.gnn_encoder_lgcl import LgclEncoder


def test_light_gcn_encoder():
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    enc = LightGCNEncoder()
    context = {
        INPUTS: [torch.as_tensor([[1.0, 2.0, 3.0]]), torch.as_tensor([[4.0, 5.0, 6]])],
        FANOUTS: [1],
    }

    expected = np.array([2.5, 3.5, 4.5])

    output = enc(context)
    npt.assert_allclose(output[0].detach().numpy(), expected, rtol=1e-4)


def test_gat_encoder():
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    enc = GatEncoder(16, 4)
    combind_feats = torch.randn(128, 2, 16)

    expected = np.array(
        [
            [-0.01335, 0.274841, -0.449927, 0.270694],
            [-0.011068, 0.227869, -0.373032, 0.224431],
        ]
    )

    output = enc(combind_feats)
    npt.assert_allclose(output[0].detach().numpy(), expected, rtol=1e-4)


def test_hetgnn_encoder():
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    lgclenc = LgclEncoder(16, [2], [16], "add", [nn.functional.leaky_relu])
    enc = HetGnnEncoder(16, [16], [[2]], [lgclenc], [nn.functional.leaky_relu])
    context = {
        INPUTS: [
            [torch.randn(4, 16), torch.randn(8, 16)],
            [torch.randn(4, 16), torch.randn(8, 16)],
        ],
        FANOUTS: [2],
    }

    expected = np.array(
        [
            -0.565202,
            1.197322,
            -2.099581,
            0.879703,
            0.049026,
            1.638405,
            -0.974986,
            -2.317848,
            2.122594,
            1.387444,
            0.810461,
            0.168435,
            -0.63819,
            -1.54342,
            -1.078809,
            0.852195,
        ]
    )

    output = enc(context)
    npt.assert_allclose(output[0].detach().numpy(), expected, rtol=1e-4)


def test_lgcl_encoder():
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    enc = LgclEncoder(16, [2], [16], "add", [nn.functional.leaky_relu])
    context = {INPUTS: [torch.randn(4, 16), torch.randn(8, 16)], FANOUTS: [2]}

    expected = np.array(
        [
            -0.565202,
            1.197322,
            -2.099581,
            0.879703,
            0.049026,
            1.638405,
            -0.974986,
            -2.317848,
            2.122594,
            1.387444,
            0.810461,
            0.168435,
            -0.63819,
            -1.54342,
            -1.078809,
            0.852195,
        ]
    )

    output = enc(context)
    npt.assert_allclose(output[0].detach().numpy(), expected, rtol=1e-4)
