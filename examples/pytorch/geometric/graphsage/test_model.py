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

from model import PTGSupervisedGraphSage  # type: ignore

from torch.utils.data import Dataset, DataLoader, Sampler

from deepgnn.graph_engine.snark.local import Client
from deepgnn import get_logger
from deepgnn.pytorch.common.utils import set_seed
from deepgnn.graph_engine.snark.converter.options import DataConverterType
from deepgnn.graph_engine.data.citation import Cora
from model import GAT, GATQueryParameter  # type: ignore
from deepgnn.graph_engine import Graph, graph_ops

class DeepGNNDataset(Dataset):
    """Cora dataset with file sampler."""
    def __init__(self, data_dir: str, node_types: List[int], feature_meta: List[int], label_meta: List[int], feature_type: np.dtype, label_type: np.dtype, neighbor_edge_types: List[int] = [0], num_hops: int = 2):
        self.g = Client(data_dir, [0, 1])
        self.node_types = np.array(node_types)
        self.feature_meta = np.array([feature_meta])
        self.label_meta = np.array([label_meta])
        self.feature_type = feature_type
        self.label_type = label_type
        self.neighbor_edge_types = np.array(neighbor_edge_types, np.int64)
        self.num_hops = num_hops
        self.count = self.g.node_count(self.node_types)

    def __len__(self):
        return self.count

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        """Query used to generate data for training."""
        inputs = np.array([idx], np.int64)
        nodes, edges, src_idx = graph_ops.sub_graph(
            self.g,
            inputs,
            edge_types=self.neighbor_edge_types,
            num_hops=self.num_hops,
            self_loop=True,
            undirected=True,
            return_edges=True,
        )
        input_mask = np.zeros(nodes.size, np.bool)
        input_mask[src_idx] = True

        feat = self.g.node_features(nodes, self.feature_meta, self.feature_type)
        label = self.g.node_features(nodes, self.label_meta, self.label_type)
        label = label.astype(np.int32)
        edges_value = np.ones(edges.shape[0], np.float32)
        edges = np.transpose(edges)
        adj_shape = np.array([nodes.size, nodes.size], np.int64)

        return nodes.reshape((1, -1)), feat.reshape((1, -1)), input_mask.reshape((1, -1)), label.reshape((1, -1)), edges.reshape((1, -1)), edges_value.reshape((1, -1)), adj_shape.reshape((1, -1))


class FileSampler(Sampler[int]):
    def __init__(self, filename: str):
        self.filename = filename

    def __len__(self) -> int:
        raise NotImplementedError("")

    def __iter__(self) -> Iterator[int]:
        with open(self.filename, "r") as file:
            for line in file.readlines():
                yield int(line)


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

    dataset = DeepGNNDataset(mock_graph, [0, 1, 2], [feature_idx, feature_dim], [label_idx, label_dim], np.float32, np.float32)

    times = []
    loss_list = []
    while True:
        trainloader = DataLoader(dataset, sampler=FileSampler(os.path.join(g.data_dir(), "train.nodes")), batch_size=256)

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
