# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import numpy as np
import numpy.testing as npt
import os
import sys
import pytest
import random
import argparse
import torch

from examples.pytorch.pyg.train import DeepGNNFeatureStore, DeepGNNGraphStore
from deepgnn.graph_engine import SamplingStrategy
from deepgnn.graph_engine.data.citation import Cora
from torch_geometric.loader import LinkNeighborLoader


def test_interface():
    np.random.seed(0)
    torch.manual_seed(0)
    random.seed(0)

    ge = Cora()
    loader = LinkNeighborLoader(
        (DeepGNNFeatureStore(ge), DeepGNNGraphStore(ge)),
        batch_size=128,
        shuffle=True,
        neg_sampling_ratio=1.0,
        num_neighbors=[10, 10],
        num_workers=1,
        persistent_workers=True,
        edge_label_index=(
            ("0", "0", "0"),
            torch.Tensor(ge.sample_edges(100, np.array(0), SamplingStrategy.Random))
            .long()[:, :2]
            .T,
        ),
    )
    data = next(iter(loader))
    assert data[("0", "0", "0")].y.shape[0] >= 1300
    assert data[("0", "0", "0")].x.shape[1] == 50
    assert data[("0", "0", "0")].y.shape[0] >= 1300
    assert data[("0", "0", "0")].y.shape[1] == 121
    assert data[("0", "0", "0")].edge_label_index.shape == (2, 200)


if __name__ == "__main__":
    sys.exit(
        pytest.main(
            [__file__, "--junitxml", os.environ["XML_OUTPUT_FILE"], *sys.argv[1:]]
        )
    )
