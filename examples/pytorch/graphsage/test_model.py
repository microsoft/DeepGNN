# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import pytest
import sys
import os
import platform
import tempfile
from typing import Dict
import numpy as np
import numpy.testing as npt
import torch
import ray
from deepgnn import get_logger
from deepgnn.pytorch.common import F1Score
from deepgnn.graph_engine.data.citation import Cora

from model import PTGSupervisedGraphSage  # type: ignore
from deepgnn.graph_engine.snark.distributed import Server, Client as DistributedClient
from deepgnn.graph_engine.data.citation import Cora

from examples.pytorch.conftest import (  # noqa: F401
    MockSimpleDataLoader,
    MockFixedSimpleDataLoader,
    mock_graph,
    prepare_local_test_files,
)


def setup_module(module):
    import deepgnn.graph_engine.snark._lib as lib

    lib_name = "libwrapper.so"
    if platform.system() == "Windows":
        lib_name = "wrapper.dll"

    os.environ[lib._SNARK_LIB_PATH_ENV_KEY] = os.path.join(
        os.path.dirname(__file__), "..", "..", "..", "src", "cc", "lib", lib_name
    )


def test_supervised_graphsage_model(mock_graph):  # noqa: F811
    np.random.seed(0)
    torch.manual_seed(0)

    num_classes = 7
    label_dim = 7
    label_idx = 1
    feature_dim = 1433
    feature_idx = 0
    edge_type = 0

    # once the model's layers and random seed are fixed, output of the input
    # is deterministic.
    nodes = torch.as_tensor([2700])
    expected = np.array(
        [[0.042573, -0.018846, 0.037796, -0.033781, -0.067056, 0.024851, 0.077601]]
    )
    graphsage = PTGSupervisedGraphSage(
        num_classes=num_classes,
        metric=F1Score(),
        label_idx=label_idx,
        label_dim=label_dim,
        feature_type=np.float32,
        feature_idx=feature_idx,
        feature_dim=feature_dim,
        edge_type=edge_type,
        fanouts=[5, 5],
    )
    simpler = MockFixedSimpleDataLoader(
        nodes.numpy(), query_fn=graphsage.query, graph=mock_graph
    )
    trainloader = torch.utils.data.DataLoader(simpler)
    it = iter(trainloader)

    query_output = it.next()
    query_expected = {
        "inputs": np.array([[2700]]),
        "label": np.array([[[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]]]),
        "out_1": np.array([[6]]),
        "out_2": np.array([[1]]),
    }
    for key in query_expected.keys():
        npt.assert_allclose(query_output[key].detach().numpy(), query_expected[key])

    output = graphsage.get_score(query_output)
    npt.assert_allclose(output.detach().numpy(), expected, rtol=1e-4)


if __name__ == "__main__":
    sys.exit(
        pytest.main(
            [__file__, "--junitxml", os.environ["XML_OUTPUT_FILE"], *sys.argv[1:]]
        )
    )
