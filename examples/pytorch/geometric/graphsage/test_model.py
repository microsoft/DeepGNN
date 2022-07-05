# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import pytest
import tempfile
import time
import numpy.testing as npt
import os
import sys
import torch
import numpy as np

from deepgnn import get_logger
from deepgnn.pytorch.common import F1Score

from examples.pytorch.conftest import (  # noqa: F401
    MockSimpleDataLoader,
    MockFixedSimpleDataLoader,
    mock_graph,
    prepare_local_test_files,
)

from model import PTGSupervisedGraphSage


@pytest.fixture(scope="module")
def train_supervised_graphsage(mock_graph):  # noqa: F811
    torch.manual_seed(0)
    np.random.seed(0)
    num_classes = 7
    label_dim = 7
    label_idx = 1
    feature_dim = 1433
    feature_idx = 0
    edge_type = 0

    model_path = tempfile.TemporaryDirectory()
    model_path_name = model_path.name + "/gnnmodel.pt"

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
        filter(lambda p: p.requires_grad, graphsage.parameters()), lr=0.7
    )
    times = []
    loss_list = []
    while True:
        trainloader = torch.utils.data.DataLoader(
            MockSimpleDataLoader(
                batch_size=256, query_fn=graphsage.query, graph=mock_graph
            )
        )

        for i, context in enumerate(trainloader):
            start_time = time.time()
            optimizer.zero_grad()
            loss, _, _ = graphsage(context)
            loss.backward()
            optimizer.step()
            end_time = time.time()
            times.append(end_time - start_time)
            get_logger().info("step: {}; loss: {} ".format(i, loss.data.item()))
            loss_list.append(loss)

            if len(times) == 100:
                break

        if len(times) == 100:
            break

    torch.save(graphsage.state_dict(), model_path_name)

    yield {"losses": loss_list, "model_path": model_path_name, "graph": mock_graph}

    model_path.cleanup()


def test_deep_graph_on_cora(train_supervised_graphsage):  # noqa: F811
    torch.manual_seed(0)
    np.random.seed(0)
    num_nodes = 2708
    num_classes = 7
    label_dim = 7
    label_idx = 1
    feature_dim = 1433
    feature_idx = 0
    edge_type = 0
    train_ctx = train_supervised_graphsage

    metric = F1Score()
    g = train_ctx["graph"]
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

    graphsage.load_state_dict(torch.load(train_ctx["model_path"]))
    graphsage.train()

    # Generate validation dataset from random indices
    rand_indices = np.random.RandomState(seed=1).permutation(num_nodes)
    val_ref = rand_indices[1000:1500]
    simpler = MockFixedSimpleDataLoader(val_ref, query_fn=graphsage.query, graph=g)
    trainloader = torch.utils.data.DataLoader(simpler)
    it = iter(trainloader)
    val_output_ref = graphsage.get_score(it.next())
    val_labels = g.node_features(
        val_ref, np.array([[label_idx, label_dim]]), np.float32
    ).argmax(1)
    f1_ref = metric.compute(val_output_ref.argmax(axis=1), val_labels)

    assert 0.80 < f1_ref and f1_ref < 0.95


# test to make sure loss decrease.
def test_supervised_graphsage_loss(train_supervised_graphsage):
    train_ctx = train_supervised_graphsage

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
