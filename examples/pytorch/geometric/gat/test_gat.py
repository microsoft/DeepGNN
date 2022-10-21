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
from deepgnn.pytorch.common.dataset import TorchDeepGNNDataset
from deepgnn.graph_engine import (
    GraphType,
    BackendType,
    FileNodeSampler,
    BackendOptions,
    create_backend,
)
from deepgnn.graph_engine.snark.converter.options import DataConverterType
from deepgnn.graph_engine.data.citation import Cora

from model_geometric import GAT, GATQueryParameter  # type: ignore
from deepgnn import get_logger


def setup_test(main_file):
    tmp_dir = tempfile.TemporaryDirectory()
    current_dir = os.path.dirname(os.path.realpath(__file__))
    mainfile = os.path.join(current_dir, main_file)
    return tmp_dir, tmp_dir.name, tmp_dir.name, mainfile


def test_pytorch_gat_cora():
    set_seed(123)
    g = Cora()
    qparam = GATQueryParameter(
        neighbor_edge_types=np.array([0], np.int32),
        feature_idx=0,
        feature_dim=g.FEATURE_DIM,
        label_idx=1,
        label_dim=1,
    )
    model = GAT(
        in_dim=g.FEATURE_DIM,
        head_num=[8, 1],
        hidden_dim=8,
        num_classes=g.NUM_CLASSES,
        ffd_drop=0.6,
        attn_drop=0.6,
        q_param=qparam,
    )

    args = argparse.Namespace(
        data_dir=g.data_dir(),
        backend=BackendType.SNARK,
        graph_type=GraphType.LOCAL,
        converter=DataConverterType.SKIP,
        partitions=[0],
    )

    backend = create_backend(BackendOptions(args), is_leader=True)

    def create_dataset():
        ds = TorchDeepGNNDataset(
            sampler_class=FileNodeSampler,
            backend=backend,
            query_fn=model.q.query_training,
            prefetch_queue_size=1,
            prefetch_worker_size=1,
            sample_files=os.path.join(g.data_dir(), "train.nodes"),
            batch_size=140,
            shuffle=True,
            drop_last=True,
            worker_index=0,
            num_workers=1,
        )

        return torch.utils.data.DataLoader(ds)

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=0.005,
        weight_decay=0.0005,
    )

    # train
    num_epochs = 200
    ds = create_dataset()
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
    def create_eval_dataset():
        ds = TorchDeepGNNDataset(
            sampler_class=FileNodeSampler,
            backend=backend,
            query_fn=model.q.query_training,
            prefetch_queue_size=1,
            prefetch_worker_size=1,
            batch_size=1000,
            sample_files=os.path.join(g.data_dir(), "test.nodes"),
            shuffle=False,
            drop_last=True,
            worker_index=0,
            num_workers=1,
        )

        return torch.utils.data.DataLoader(ds)

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
