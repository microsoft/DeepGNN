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

from deepgnn import get_logger
from deepgnn.pytorch.common import F1Score
from deepgnn.graph_engine.data.citation import Cora

from main import run_ray  # type: ignore
from model import PTGSupervisedGraphSage  # type: ignore

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


@pytest.fixture(scope="module")
def train_graphsage_cora_ddp_trainer(mock_graph):
    model_dir = tempfile.TemporaryDirectory()
    working_dir = tempfile.TemporaryDirectory()
    Cora(working_dir.name)

    def create_mock_dataset(
        args,
        model,
        rank: int = 0,
        world_size: int = 1,
        address: str=None,
    ):
        dataset = MockSimpleDataLoader(
            batch_size=256, query_fn=model.query, graph=mock_graph
        )
        num_workers = (
            0
            if platform.system() == "Windows"
            else args.data_parallel_num
        )
        dataset = torch.utils.data.DataLoader(
            dataset=dataset,
            num_workers=num_workers,
        )
        return dataset

    result = run_ray(
        init_dataset_fn=create_mock_dataset,
        run_args=[
            "--data_dir",
            working_dir.name,
            "--mode",
            "train",
            "--seed",
            "123",
            "--backend",
            "snark",
            "--graph_type",
            "local",
            "--converter skip",
            "--batch_size",
            "256",
            "--learning_rate",
            "0.7",
            "--num_epochs",
            "10",
            "--node_type",
            "0",
            "--max_id",
            "-1",
            "--model_dir",
            model_dir.name,
            "--metric_dir",
            model_dir.name,
            "--save_path",
            model_dir.name,
            "--feature_idx",
            "0",
            "--feature_dim",
            "1433",
            "--label_idx",
            "1",
            "--label_dim",
            "7",
            "--algo",
            "supervised",
        ],
    )
    yield {
        "losses": result.metrics["losses"],
        "model_path": os.path.join(model_dir.name, "gnnmodel-008-000007.pt"),
    }
    working_dir.cleanup()
    model_dir.cleanup()


def test_deep_graph_on_cora(train_graphsage_cora_ddp_trainer, mock_graph):  # noqa: F811
    torch.manual_seed(0)
    np.random.seed(0)
    num_nodes = 2708
    num_classes = 7
    label_dim = 7
    label_idx = 1
    feature_dim = 1433
    feature_idx = 0
    edge_type = 0
    train_ctx = train_graphsage_cora_ddp_trainer

    metric = F1Score()
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

    graphsage.load_state_dict(torch.load(train_ctx["model_path"])["state_dict"])
    graphsage.train()

    # Generate validation dataset from random indices
    g = mock_graph
    rand_indices = np.random.RandomState(seed=1).permutation(num_nodes)
    val_ref = rand_indices[1000:1500]
    simpler = MockFixedSimpleDataLoader(val_ref, query_fn=graphsage.query, graph=g)
    trainloader = torch.utils.data.DataLoader(simpler)
    it = iter(trainloader)
    val_output_ref = graphsage.get_score(it.next())
    val_labels = g.node_features(
        val_ref, np.array([[label_idx, label_dim]]), np.float32
    ).argmax(1)
    # assert False, (val_labels, val_output_ref.argmax(axis=1))
    f1_ref = metric.compute(val_output_ref.argmax(axis=1), val_labels)

    assert 0.80 < f1_ref and f1_ref < 0.95


# test to make sure loss decrease.
def test_supervised_graphsage_loss(train_graphsage_cora_ddp_trainer):
    train_ctx = train_graphsage_cora_ddp_trainer

    # make sure the loss decreased in training.
    assert len(train_ctx["losses"]) > 0
    assert train_ctx["losses"][len(train_ctx["losses"]) - 1] < train_ctx["losses"][0]


# fix the seeds to test the algo's correctness.
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
    output = graphsage.get_score(it.next())
    npt.assert_allclose(output.detach().numpy(), expected, rtol=1e-4)


# test the correctness of the loss function.
def test_supervised_graphsage_loss_value(mock_graph):  # noqa: F811
    np.random.seed(0)
    torch.manual_seed(0)

    num_classes = 7
    label_dim = 7
    label_idx = 1
    feature_dim = 1433
    feature_idx = 0
    edge_type = 0

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
    optimizer = torch.optim.SGD(
        filter(lambda p: p.requires_grad, graphsage.parameters()), lr=0.01
    )

    # use one batch to verify the output loss value.
    trainloader = torch.utils.data.DataLoader(
        MockSimpleDataLoader(batch_size=256, query_fn=graphsage.query, graph=mock_graph)
    )
    it = iter(trainloader)
    optimizer.zero_grad()
    loss, _, _ = graphsage(it.next())
    loss.backward()
    optimizer.step()
    npt.assert_allclose(loss.detach().numpy(), np.array([1.941329]), rtol=1e-5)


if __name__ == "__main__":
    sys.exit(
        pytest.main(
            [__file__, "--junitxml", os.environ["XML_OUTPUT_FILE"], *sys.argv[1:]]
        )
    )
