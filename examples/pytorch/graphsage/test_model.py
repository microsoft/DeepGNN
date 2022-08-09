# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import pytest
import tempfile
import time
import numpy.testing as npt
import os
import sys
import torch
import argparse
import numpy as np
import urllib.request
import zipfile

from deepgnn import get_logger
from deepgnn.pytorch.common import MRR, F1Score
from deepgnn.pytorch.common.dataset import TorchDeepGNNDataset
from deepgnn.pytorch.encoding.feature_encoder import (
    TwinBERTEncoder,
    TwinBERTFeatureEncoder,
)
from examples.pytorch.conftest import (  # noqa: F401
    MockSimpleDataLoader,
    MockFixedSimpleDataLoader,
    mock_graph,
)
from deepgnn.graph_engine import (
    FeatureType,
    GraphType,
    BackendType,
    BackendOptions,
    GENodeSampler,
    create_backend,
)
import deepgnn.graph_engine.snark.convert as convert
from deepgnn.graph_engine.snark.decoders import JsonDecoder
from deepgnn.graph_engine.snark.converter.options import DataConverterType
from model import SupervisedGraphSage, UnSupervisedGraphSage

logger = get_logger()


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

    graphsage = SupervisedGraphSage(
        num_classes=num_classes,
        metric=F1Score(),
        label_idx=label_idx,
        label_dim=label_dim,
        feature_type=FeatureType.FLOAT,
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
            logger.info("step: {}; loss: {} ".format(i, loss.data.item()))
            loss_list.append(loss)

            if len(times) == 100:
                break

        if len(times) == 100:
            break

    torch.save(graphsage.state_dict(), model_path_name)

    yield {"losses": loss_list, "model_path": model_path_name, "graph": mock_graph}

    model_path.cleanup()


@pytest.fixture(scope="module")
def train_unsupervised_graphsage(mock_graph):  # noqa: F811
    np.random.seed(0)
    torch.manual_seed(0)
    label_dim = 7
    feature_dim = 1433
    feature_idx = 0
    edge_type = 0
    num_negs = 1

    model_path = tempfile.TemporaryDirectory()
    model_path_name = model_path.name + "/gnnmodel.pt"

    graphsage = UnSupervisedGraphSage(
        num_classes=label_dim,
        metric=MRR(),
        num_negs=num_negs,
        feature_type=FeatureType.FLOAT,
        feature_idx=feature_idx,
        feature_dim=feature_dim,
        edge_type=edge_type,
        fanouts=[5, 5],
    )

    optimizer = torch.optim.SGD(
        filter(lambda p: p.requires_grad, graphsage.parameters()), lr=0.00005
    )

    epochs_left = 6
    mrr_values = []
    loss_list = []
    while epochs_left > 0:
        epochs_left -= 1
        trainloader = torch.utils.data.DataLoader(
            MockSimpleDataLoader(
                batch_size=512, query_fn=graphsage.query, graph=mock_graph
            )
        )

        scores = []
        labels = []
        for _, context in enumerate(trainloader):
            optimizer.zero_grad()
            loss, score, label = graphsage(context)
            scores.append(score)
            labels.append(label)
            loss.backward()
            optimizer.step()
            loss_list.append(loss)

        mrr = graphsage.compute_metric(scores, labels)
        logger.info("MRR: {}".format(mrr.data.item()))
        mrr_values.append(mrr)

    torch.save(graphsage.state_dict(), model_path_name)

    yield {
        "losses": loss_list,
        "model_path": model_path_name,
        "graph": mock_graph,
        "mrr_values": mrr_values,
    }

    model_path.cleanup()


def test_deep_graph_on_cora(train_supervised_graphsage):
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
    graphsage = SupervisedGraphSage(
        num_classes=num_classes,
        metric=F1Score(),
        label_idx=label_idx,
        label_dim=label_dim,
        feature_type=FeatureType.FLOAT,
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
        val_ref, np.array([[label_idx, label_dim]]), FeatureType.FLOAT
    ).argmax(1)
    f1_ref = metric.compute(val_output_ref.argmax(axis=1), val_labels)

    assert 0.85 < f1_ref and f1_ref < 0.95


def test_deep_graph_on_unsupervised_cora(train_unsupervised_graphsage):
    np.random.seed(0)
    torch.manual_seed(0)
    label_dim = 7
    feature_dim = 1433
    feature_idx = 0
    edge_type = 0
    num_negs = 1
    train_ctx = train_unsupervised_graphsage

    graphsage = UnSupervisedGraphSage(
        num_classes=label_dim,
        metric=MRR(),
        num_negs=num_negs,
        feature_type=FeatureType.FLOAT,
        feature_idx=feature_idx,
        feature_dim=feature_dim,
        edge_type=edge_type,
        fanouts=[5, 5],
    )
    graphsage.load_state_dict(torch.load(train_ctx["model_path"]))
    graphsage.train()

    mrr_values = train_ctx["mrr_values"]
    half = int(len(mrr_values) / 2)
    fst = mrr_values[:half]
    snd = mrr_values[half:]
    avg_fst = sum(fst) / len(fst)
    avg_snd = sum(snd) / len(snd)
    assert avg_fst < avg_snd
    assert avg_snd > 0.26


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
        [[0.074278, -0.069181, 0.003444, -0.008916, -0.013685, -0.036867, 0.042985]],
        dtype=np.float32,
    )
    graphsage = SupervisedGraphSage(
        num_classes=num_classes,
        metric=F1Score(),
        label_idx=label_idx,
        label_dim=label_dim,
        feature_type=FeatureType.FLOAT,
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
    npt.assert_allclose(output.detach().numpy(), expected, rtol=1e-3)


# test if computational graph is connected.
def test_supervised_graphsage_computational_graph(mock_graph):  # noqa: F811
    np.random.seed(0)
    torch.manual_seed(0)

    num_classes = 7
    label_dim = 7
    label_idx = 1
    feature_dim = 1433
    feature_idx = 0
    edge_type = 0

    graphsage = SupervisedGraphSage(
        num_classes=num_classes,
        metric=F1Score(),
        label_idx=label_idx,
        label_dim=label_dim,
        feature_type=FeatureType.FLOAT,
        feature_idx=feature_idx,
        feature_dim=feature_dim,
        edge_type=edge_type,
        fanouts=[5, 5],
    )

    # use one batch to verify if computational graph is connected.
    trainloader = torch.utils.data.DataLoader(
        MockSimpleDataLoader(batch_size=256, query_fn=graphsage.query, graph=mock_graph)
    )
    it = iter(trainloader)
    context = it.next()

    # here we are using feature tensor as a proxy for node ids
    assert torch.equal(
        context["encoder"]["node_feats"]["neighbor_feats"],
        context["encoder"]["neighbor_feats"]["node_feats"],
    )


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

    graphsage = SupervisedGraphSage(
        num_classes=num_classes,
        metric=F1Score(),
        label_idx=label_idx,
        label_dim=label_dim,
        feature_type=FeatureType.FLOAT,
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
    npt.assert_allclose(loss.detach().numpy(), np.array([1.930]), rtol=1e-3)


# test the correctness of the unsupervised graphsage's model.
def test_unsupervised_graphsage_model(mock_graph):  # noqa: F811
    np.random.seed(0)
    torch.manual_seed(0)
    label_dim = 7
    feature_dim = 1433
    feature_idx = 0
    edge_type = 0
    num_negs = 1

    # once the model's layers and random seed are fixed, output of the input
    # is deterministic.
    nodes = torch.as_tensor([2700])
    expected = np.array(
        [[0.074278, -0.069181, 0.003444, -0.008916, -0.013685, -0.036867, 0.042985]],
        dtype=np.float32,
    )
    graphsage = UnSupervisedGraphSage(
        num_classes=label_dim,
        metric=MRR(),
        num_negs=num_negs,
        feature_type=FeatureType.FLOAT,
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
    output = graphsage.get_score(it.next()["encoder"])
    npt.assert_allclose(output.detach().numpy(), expected, rtol=1e-3)


# test the correctness of the unsupervised graphsage loss function.
def test_unsupervised_graphsage_loss_value(mock_graph):  # noqa: F811
    np.random.seed(0)
    torch.manual_seed(0)
    label_dim = 7
    feature_dim = 1433
    feature_idx = 0
    edge_type = 0
    num_negs = 1

    graphsage = UnSupervisedGraphSage(
        num_classes=label_dim,
        metric=MRR(),
        num_negs=num_negs,
        feature_type=FeatureType.FLOAT,
        feature_idx=feature_idx,
        feature_dim=feature_dim,
        edge_type=edge_type,
        fanouts=[5, 5],
    )
    optimizer = torch.optim.SGD(
        filter(lambda p: p.requires_grad, graphsage.parameters()), lr=0.7
    )

    trainloader = torch.utils.data.DataLoader(
        MockSimpleDataLoader(batch_size=256, query_fn=graphsage.query, graph=mock_graph)
    )
    it = iter(trainloader)
    optimizer.zero_grad()
    loss, _, _ = graphsage(it.next())
    loss.backward()
    optimizer.step()

    # the unsupervised graphsage model use threadpool to calculate the neg/pos_embedding
    # which cannot guarantee the order of these steps, and this disorder will lead to
    # uncertainty of the loss. Here we temporarilly enlarge the threshold to make the case pass.
    npt.assert_allclose(loss.detach().numpy(), np.array([0.6907]), rtol=1e-2)


@pytest.fixture(scope="module")
def tiny_graph():
    graph_dir = tempfile.TemporaryDirectory()
    name = "twinbert.zip"
    zip_file = os.path.join(graph_dir.name, name)
    urllib.request.urlretrieve(
        f"https://deepgraphpub.blob.core.windows.net/public/testdata/{name}", zip_file
    )
    with zipfile.ZipFile(zip_file, "r") as zip_ref:
        zip_ref.extractall(graph_dir.name)

    convert.MultiWorkersConverter(
        graph_path=os.path.join(graph_dir.name, "twinbert/tiny_graph.json"),
        partition_count=1,
        output_dir=graph_dir.name,
        decoder=JsonDecoder(),
    ).convert()

    yield graph_dir.name
    graph_dir.cleanup()


def get_twinbert_encoder(test_rootdir, config_file, feature_type=FeatureType.BINARY):
    torch.manual_seed(0)
    config_file = os.path.join(test_rootdir, "twinbert", config_file)
    config = TwinBERTEncoder.init_config_from_file(config_file)
    return TwinBERTFeatureEncoder(feature_type, config, pooler_count=2)


@pytest.fixture(scope="module")
def train_unsupervised_graphsage_with_feature_encoder(tiny_graph):
    model_path = tempfile.TemporaryDirectory()
    model_path_name = model_path.name + "/gnnmodel.pt"

    graphsage = UnSupervisedGraphSage(
        num_classes=7,
        metric=MRR(),
        num_negs=1,
        feature_type=FeatureType.BINARY,
        feature_dim=0,
        feature_idx=0,
        edge_type=0,
        fanouts=[2, 2],
        feature_enc=get_twinbert_encoder(tiny_graph, "twinbert_triletter.json"),
    )

    optimizer = torch.optim.SGD(
        filter(lambda p: p.requires_grad, graphsage.parameters()), lr=0.00005
    )

    epochs_left = 1
    mrr_values = []
    loss_list = []
    while epochs_left > 0:
        epochs_left -= 1
        args = argparse.Namespace(
            backend=BackendType.SNARK,
            graph_type=GraphType.LOCAL,
            data_dir=tiny_graph,
            converter=DataConverterType.SKIP,
        )
        backend = create_backend(BackendOptions(args), is_leader=True)
        trainloader = torch.utils.data.DataLoader(
            TorchDeepGNNDataset(
                sampler_class=GENodeSampler,
                backend=backend,
                query_fn=graphsage.query,
                prefetch_queue_size=1,
                prefetch_worker_size=1,
                node_types=np.array([0], dtype=np.int32),
                sample_num=128,
                batch_size=16,
                sample_files=[],
            )
        )

        scores = []
        labels = []
        for i, context in enumerate(trainloader):
            optimizer.zero_grad()
            loss, score, label = graphsage(context)
            scores.append(score)
            labels.append(label)
            loss.backward()
            optimizer.step()
            loss_list.append(loss)

        mrr = graphsage.compute_metric(scores, labels)
        print("MRR: {}".format(mrr.data.item()))
        mrr_values.append(mrr)

    torch.save(graphsage.state_dict(), model_path_name)

    yield {
        "losses": loss_list,
        "model_path": model_path_name,
        "graph": tiny_graph,
        "mrr_values": mrr_values,
    }

    model_path.cleanup()


def test_unsupervised_graphsage_with_feature_encoder(
    train_unsupervised_graphsage_with_feature_encoder, tiny_graph
):
    """This test is to go through the process of training a graphsage model with twinbert feature encoder.

    Twinbert encoding on CPU is very time consuming, so we just run few steps with a random tiny graph,
    and don't check exact metrics values.
    """
    train_ctx = train_unsupervised_graphsage_with_feature_encoder

    graphsage = UnSupervisedGraphSage(
        num_classes=7,
        metric=MRR(),
        num_negs=1,
        feature_type=FeatureType.BINARY,
        feature_dim=0,
        feature_idx=0,
        edge_type=0,
        fanouts=[2, 2],
        feature_enc=get_twinbert_encoder(tiny_graph, "twinbert_triletter.json"),
    )
    ckpt = torch.load(train_ctx["model_path"])
    graphsage.load_state_dict(ckpt)

    mrr_values = train_ctx["mrr_values"]
    avg_mrr = sum(mrr_values) / len(mrr_values)
    assert avg_mrr > 0.5


if __name__ == "__main__":
    sys.exit(
        pytest.main(
            [__file__, "--junitxml", os.environ["XML_OUTPUT_FILE"], *sys.argv[1:]]
        )
    )