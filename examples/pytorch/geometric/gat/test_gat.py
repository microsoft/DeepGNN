# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import sys
import pytest
import tempfile
import numpy as np
import torch
import argparse

from deepgnn.pytorch.common.utils import set_seed
from deepgnn.graph_engine.snark.converter.options import DataConverterType
from deepgnn.graph_engine.data.citation import Cora

from model_geometric import GAT, GATQueryParameter  # type: ignore
from deepgnn import get_logger

from deepgnn.graph_engine.snark.local import Client
from deepgnn import get_logger
from deepgnn.pytorch.common.utils import set_seed
from deepgnn.graph_engine.snark.converter.options import DataConverterType
from deepgnn.graph_engine.data.citation import Cora
from deepgnn.graph_engine import Graph, graph_ops


def setup_test(main_file):
    tmp_dir = tempfile.TemporaryDirectory()
    current_dir = os.path.dirname(os.path.realpath(__file__))
    mainfile = os.path.join(current_dir, main_file)
    return tmp_dir, tmp_dir.name, tmp_dir.name, mainfile


def test_pytorch_gat_cora():
    set_seed(123)
    g = Cora()
    model = GAT(
        in_dim=g.FEATURE_DIM,
        head_num=[8, 1],
        hidden_dim=8,
        num_classes=g.NUM_CLASSES,
        ffd_drop=0.6,
        attn_drop=0.6,
    )

    dataset = GATGeoDataset(g.data_dir(), [0, 1, 2], [0, g.FEATURE_DIM], [1, 1], np.float32, np.float32)

    ds = DataLoader(dataset, sampler=BatchedSampler(FileNodeSampler(os.path.join(g.data_dir(), "train.nodes")), 140))

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=0.005,
        weight_decay=0.0005,
    )

    # train
    num_epochs = 200
    model.train()
    for ei in range(num_epochs):
        for si, batch_input in enumerate(ds):
            optimizer.zero_grad()
            loss, pred, label = model(batch_input)
            loss.backward()
            optimizer.step()
            acc = model.compute_metric([pred], [label])
            get_logger().info(
                f"epoch {ei} - {si}, loss {loss.data.item() :.6f}, accuracy {acc.data.item():.6f}"
            )

    # evaluate
    test_dataset = DataLoader(dataset, sampler=BatchedSampler(FileNodeSampler(os.path.join(g.data_dir(), "test.nodes")), 1000))

    test_dataset = create_eval_dataset()
    model.eval()
    for si, batch_input in enumerate(test_dataset):
        loss, pred, label = model(batch_input)
        acc = model.compute_metric([pred], [label])
        get_logger().info(
            f"evaluate loss {loss.data.item(): .6f}, accuracy {acc.data.item(): .6f}"
        )
        np.testing.assert_allclose(acc.data.item(), 0.83, atol=0.005)


if __name__ == "__main__":
    sys.exit(
        pytest.main(
            [__file__, "--junitxml", os.environ["XML_OUTPUT_FILE"], *sys.argv[1:]]
        )
    )
