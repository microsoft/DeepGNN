# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
import os
import sys
import pytest
import random
import tempfile
import time
import numpy as np
import numpy.testing as npt
import urllib.request
import zipfile
import torch

from deepgnn import get_logger
from deepgnn.pytorch.common.consts import (
    NODE_FEATURES,
    ENCODER_SEQ,
    ENCODER_MASK,
    ENCODER_TYPES,
    FANOUTS,
    NODE_SRC,
    EMBS_LIST,
)
from examples.pytorch.link_prediction.consts import ENCODER_LABEL, FANOUTS_NAME
from deepgnn.pytorch.encoding import TwinBERTEncoder, MultiTypeFeatureEncoder
from examples.pytorch.conftest import MockGraph, load_data  # noqa: F401
from deepgnn.pytorch.common.dataset import TorchDeepGNNDataset
from deepgnn.graph_engine import (
    FeatureType,
    GraphType,
    BackendType,
    TextFileSampler,
    BackendOptions,
    GraphEngineBackend,
    create_backend,
)
from deepgnn.graph_engine.snark.converter.options import DataConverterType
from args import init_args  # type: ignore
from model import LinkPredictionModel  # type: ignore
from deepgnn.graph_engine.test_adl_reader import IS_ADL_CONFIG_VALID

pytestmark = pytest.mark.skipif(not IS_ADL_CONFIG_VALID, reason="Invalid adl config.")


class MockBackend(GraphEngineBackend):
    _backend = None

    def __new__(cls, options=None, is_leader: bool = False):
        if MockBackend._backend is None:
            MockBackend._backend = object.__new__(cls)
            np.random.RandomState(seed=1)
            MockBackend._backend._graph = None  # type: ignore

        return MockBackend._backend

    @property
    def graph(self):
        return self._graph


@pytest.fixture(scope="module")
def lp_mock_graph(prepare_local_test_files):  # noqa: F811
    feat_data, labels, adj_lists = load_data(prepare_local_test_files, 24)
    assert len(adj_lists) == 2708
    assert len(labels) == 2708
    assert len(feat_data) == 2708
    mock = MockBackend(None, False)
    mock._graph = MockGraph(feat_data, labels, adj_lists)
    return mock.graph


@pytest.fixture(scope="module")
def prepare_local_twinbert_test_files():
    name = "twinbert.zip"
    working_dir = tempfile.TemporaryDirectory()
    zip_file = os.path.join(working_dir.name, name)
    urllib.request.urlretrieve(
        f"https://deepgraphpub.blob.core.windows.net/public/testdata/{name}", zip_file
    )
    with zipfile.ZipFile(zip_file, "r") as zip_ref:
        zip_ref.extractall(working_dir.name)

    yield working_dir.name
    working_dir.cleanup()


def prepare_params(test_rootdir):
    parser = argparse.ArgumentParser()
    init_test_args(parser)
    init_args(parser)
    params, _ = parser.parse_known_args(
        [
            "--sim_type",
            "cosine",
            "train_file_dir",
            "adl://snrgnndls.azuredatalakestore.net/test_twinbert/folder02/*.txt",
        ]
    )
    setattr(params, "store_name", "snrgnndls")
    setattr(params, "filename_pattern", "/test_twinbert/folder02/*.txt")
    setattr(params, "grad_clipping", 1.0)
    setattr(params, "num_negs", 0)
    setattr(params, "feature_dim", 24)
    params.featenc_config = os.path.join(
        test_rootdir, "twinbert", "linkprediction.json"
    )
    config = TwinBERTEncoder.init_config_from_file(params.featenc_config)

    return params, config


def train_linkprediction_model(params, config, model_path, lp_mock_graph):
    num_epoch = 1
    model_path_name = os.path.join(model_path, f"gnnmodel_{params.gnn_encoder}.pt")

    feature_enc = MultiTypeFeatureEncoder(
        FeatureType.INT64, config, ["q", "k", "s"], False
    )

    lp = LinkPredictionModel(
        args=params, feature_type=FeatureType.INT64, feature_enc=feature_enc
    )

    optimizer = torch.optim.SGD(
        filter(lambda p: p.requires_grad, lp.parameters()),
        lr=0.01,
        momentum=0.5,
        weight_decay=0.01,
    )

    times = []
    loss_list = []

    args = argparse.Namespace(
        data_dir="/mock/doesnt/need/physical/path",
        backend=BackendType.CUSTOM,
        graph_type=GraphType.LOCAL,
        converter=DataConverterType.SKIP,
        custom_backendclass=MockBackend,
    )
    backend = create_backend(BackendOptions(args), is_leader=True)
    for x in range(num_epoch):
        dataset = TorchDeepGNNDataset(
            sampler_class=TextFileSampler,
            backend=backend,
            query_fn=lp.query,
            prefetch_queue_size=1,
            prefetch_worker_size=1,
            batch_size=32,
            store_name=params.store_name,
            filename=params.filename_pattern,
            shuffle=False,
            drop_last=False,
            worker_index=0,
            num_workers=1,
            epochs=1,
            buffer_size=1024,
        )

        trainloader = torch.utils.data.DataLoader(dataset)

        for i, context in enumerate(trainloader):
            start_time = time.time()
            optimizer.zero_grad()
            loss, _, _ = lp(context)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(lp.parameters(), params.grad_clipping)
            optimizer.step()
            end_time = time.time()
            times.append(end_time - start_time)
            get_logger().info("step: {}; loss: {} ".format(i, loss.data.item()))
            loss_list.append(loss)

    torch.save(lp.state_dict(), model_path_name)

    return loss_list, model_path_name


@pytest.fixture(scope="module")
def train_linkprediction_model_gat(lp_mock_graph, prepare_local_twinbert_test_files):
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    params, config = prepare_params(prepare_local_twinbert_test_files)
    params.gnn_encoder = "gat"
    model_path = tempfile.TemporaryDirectory()
    loss_list, model_path_name = train_linkprediction_model(
        params, config, model_path.name, lp_mock_graph
    )

    yield {
        "losses": loss_list,
        "model_path": model_path_name,
        "graph": lp_mock_graph,
        "params": params,
        "config": config,
    }

    model_path.cleanup()


@pytest.fixture(scope="module")
def train_linkprediction_model_hetgnn(lp_mock_graph, prepare_local_twinbert_test_files):
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    params, config = prepare_params(prepare_local_twinbert_test_files)
    params.gnn_encoder = "hetgnn_lgcl"
    model_path = tempfile.TemporaryDirectory()
    loss_list, model_path_name = train_linkprediction_model(
        params, config, model_path.name, lp_mock_graph
    )

    yield {
        "losses": loss_list,
        "model_path": model_path_name,
        "graph": lp_mock_graph,
        "params": params,
        "config": config,
    }

    model_path.cleanup()


def init_test_args(parser):
    parser.add_argument("--meta_dir", default="", type=str)
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "evaluate", "inference"],
        help="Run mode.",
    )


# fix the seeds to test the algo's correctness.
def test_link_prediction_model_gat(train_linkprediction_model_gat):
    np.random.seed(0)
    torch.manual_seed(0)
    random.seed(0)

    expected = np.array(
        [
            -0.00062489,
            0.12245575,
            0.00485472,
            0.04524879,
            0.13526294,
            0.08879019,
            0.06979912,
            0.08412716,
            0.07954618,
            0.01753873,
        ]
    )

    params = train_linkprediction_model_gat["params"]
    config = train_linkprediction_model_gat["config"]
    feature_enc = MultiTypeFeatureEncoder(
        FeatureType.INT64, config, ["q", "k", "s"], False
    )

    lp = LinkPredictionModel(
        args=params, feature_type=FeatureType.INT64, feature_enc=feature_enc
    )

    args = argparse.Namespace(
        data_dir="/mock/doesnt/need/physical/path",
        backend=BackendType.CUSTOM,
        graph_type=GraphType.LOCAL,
        converter=DataConverterType.SKIP,
        custom_backendclass=MockBackend,
    )
    backend = create_backend(BackendOptions(args), is_leader=True)
    dataset = TorchDeepGNNDataset(
        sampler_class=TextFileSampler,
        backend=backend,
        query_fn=lp.query,
        prefetch_queue_size=1,
        prefetch_worker_size=1,
        batch_size=32,
        store_name=params.store_name,
        filename=params.filename_pattern,
        shuffle=False,
        drop_last=False,
        worker_index=0,
        num_workers=1,
        epochs=1,
        buffer_size=1024,
    )

    trainloader = torch.utils.data.DataLoader(dataset)
    output = None
    for i, data in enumerate(trainloader):
        if i == 0:
            x_batch = data[NODE_FEATURES]
            context = {
                ENCODER_SEQ: x_batch[1][0],
                ENCODER_MASK: x_batch[1][1],
                ENCODER_LABEL: x_batch[3],
                ENCODER_TYPES: params.src_encoders,
                FANOUTS: params.src_fanouts,
                FANOUTS_NAME: NODE_SRC,
            }

            output = lp.get_score(context)

    # just compare the first 10 values.
    npt.assert_allclose(output[0].detach().numpy()[0][0:10], expected, rtol=1e-4)


def test_link_prediction_model_hetgnn(train_linkprediction_model_hetgnn):
    np.random.seed(0)
    torch.manual_seed(0)
    random.seed(0)

    expected = np.array(
        [
            -0.106575,
            0.182852,
            -0.120735,
            -0.010538,
            0.022652,
            0.151075,
            0.125764,
            -0.053967,
            0.124891,
            -0.419412,
        ]
    )

    params = train_linkprediction_model_hetgnn["params"]
    config = train_linkprediction_model_hetgnn["config"]
    feature_enc = MultiTypeFeatureEncoder(
        FeatureType.INT64, config, ["q", "k", "s"], False
    )
    params.gnn_encoder = "hetgnn_lgcl"

    lp = LinkPredictionModel(
        args=params, feature_type=FeatureType.INT64, feature_enc=feature_enc
    )

    args = argparse.Namespace(
        data_dir="/mock/doesnt/need/physical/path",
        backend=BackendType.CUSTOM,
        graph_type=GraphType.LOCAL,
        converter=DataConverterType.SKIP,
        custom_backendclass=MockBackend,
    )
    backend = create_backend(BackendOptions(args), is_leader=True)
    dataset = TorchDeepGNNDataset(
        sampler_class=TextFileSampler,
        backend=backend,
        query_fn=lp.query,
        prefetch_queue_size=1,
        prefetch_worker_size=1,
        batch_size=32,
        store_name=params.store_name,
        filename=params.filename_pattern,
        shuffle=False,
        drop_last=False,
        worker_index=0,
        num_workers=1,
        epochs=1,
        buffer_size=1024,
    )

    trainloader = torch.utils.data.DataLoader(dataset)
    output = None
    for i, data in enumerate(trainloader):
        if i == 0:
            x_batch = data[NODE_FEATURES]
            context = {
                ENCODER_SEQ: x_batch[1][0],
                ENCODER_MASK: x_batch[1][1],
                ENCODER_LABEL: x_batch[3],
                ENCODER_TYPES: params.src_encoders,
                FANOUTS: params.src_fanouts,
                FANOUTS_NAME: NODE_SRC,
            }

            output = lp.get_score(context)

    # just compare the first 10 values.
    npt.assert_allclose(output[0].detach().numpy()[0][0:10], expected, rtol=1e-4)


def do_inference(params, config, mock_graph, model_path):
    feature_enc = MultiTypeFeatureEncoder(
        FeatureType.INT64, config, ["q", "k", "s"], False
    )

    lp = LinkPredictionModel(
        args=params, feature_type=FeatureType.INT64, feature_enc=feature_enc
    )

    lp.load_state_dict(torch.load(model_path))
    lp.eval()

    args = argparse.Namespace(
        data_dir="/mock/doesnt/need/physical/path",
        backend=BackendType.CUSTOM,
        graph_type=GraphType.LOCAL,
        converter=DataConverterType.SKIP,
        custom_backendclass=MockBackend,
    )
    backend = create_backend(BackendOptions(args), is_leader=True)
    dataset = TorchDeepGNNDataset(
        sampler_class=TextFileSampler,
        backend=backend,
        query_fn=lp.query,
        prefetch_queue_size=1,
        prefetch_worker_size=1,
        batch_size=32,
        store_name=params.store_name,
        filename=params.filename_pattern,
        shuffle=False,
        drop_last=False,
        worker_index=0,
        num_workers=1,
        epochs=1,
        buffer_size=1024,
    )

    data_loader = torch.utils.data.DataLoader(dataset)
    results = []

    for i, context in enumerate(data_loader):
        out_temp = lp.get_embedding(context)
        results.append(out_temp)

    return results


def test_linkprediction_inference_gat(train_linkprediction_model_gat):
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    results = do_inference(
        train_linkprediction_model_gat["params"],
        train_linkprediction_model_gat["config"],
        train_linkprediction_model_gat["graph"],
        train_linkprediction_model_gat["model_path"],
    )

    expected = np.array([-0.54478, -0.552999, -0.556599])

    # just compare the first 3 scores.
    npt.assert_allclose(
        results[0][0][0:3].detach().numpy().reshape(-1), expected, rtol=1e-4
    )


def test_linkprediction_inference_hetgnn(train_linkprediction_model_hetgnn):
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    results = do_inference(
        train_linkprediction_model_hetgnn["params"],
        train_linkprediction_model_hetgnn["config"],
        train_linkprediction_model_hetgnn["graph"],
        train_linkprediction_model_hetgnn["model_path"],
    )

    expected = np.array([0.494424, 0.411162, 0.414233])

    # just compare the first 3 scores.
    npt.assert_allclose(
        results[0][0][0:3].detach().numpy().reshape(-1), expected, rtol=1e-4
    )


def test_linkprediction_query(lp_mock_graph, prepare_local_twinbert_test_files):
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    mock_graph = lp_mock_graph
    params, config = prepare_params(prepare_local_twinbert_test_files)
    feature_enc = MultiTypeFeatureEncoder(
        FeatureType.INT64, config, ["q", "k", "s"], False
    )
    lp = LinkPredictionModel(
        args=params, feature_type=FeatureType.INT64, feature_enc=feature_enc
    )

    sampler = TextFileSampler(
        batch_size=1,
        store_name="snrgnndls",
        filename="/test_twinbert/folder02/*.txt",
        shuffle=False,
        worker_index=0,
        num_workers=1,
    )

    it = iter(sampler)
    inputs = next(it)

    context = lp.query(mock_graph, inputs)
    row_id = context[NODE_FEATURES][0]
    src_seq = np.array(context[NODE_FEATURES][1][0][0])
    dst_seq = np.array(context[NODE_FEATURES][2][0][0])

    exp_row_id = ["0"]
    assert len(row_id) == 1 and row_id[0] == exp_row_id[0]

    exp_seq = np.array([71, 918, 27835, 21190, 0, 0, 0, 0, 0, 0])
    npt.assert_allclose(src_seq.reshape(-1)[0:10], exp_seq, rtol=1e-4)

    exp_dst_seq = np.array([27835, 21190, 313, 27835, 21190, 8, 96, 1238, 1103, 7])
    npt.assert_allclose(dst_seq.reshape(-1)[0:10], exp_dst_seq, rtol=1e-4)


def run_multitype_feature_encoder(params, config, graph, shared=False):
    node_types = ["q", "k", "s"]
    enc = MultiTypeFeatureEncoder(FeatureType.INT64, config, node_types, shared)

    lp = LinkPredictionModel(
        args=params, feature_type=FeatureType.INT64, feature_enc=enc
    )

    args = argparse.Namespace(
        data_dir="/mock/doesnt/need/physical/path",
        backend=BackendType.CUSTOM,
        graph_type=GraphType.LOCAL,
        converter=DataConverterType.SKIP,
        custom_backendclass=MockBackend,
    )
    backend = create_backend(BackendOptions(args), is_leader=True)
    dataset = TorchDeepGNNDataset(
        sampler_class=TextFileSampler,
        backend=backend,
        query_fn=lp.query,
        prefetch_queue_size=1,
        prefetch_worker_size=1,
        batch_size=32,
        store_name=params.store_name,
        filename=params.filename_pattern,
        shuffle=False,
        drop_last=False,
        worker_index=0,
        num_workers=1,
        epochs=1,
        buffer_size=1024,
    )

    trainloader = torch.utils.data.DataLoader(dataset)
    output = None
    for i, data in enumerate(trainloader):
        if i == 0:
            x_batch = data[NODE_FEATURES]
            context = {
                ENCODER_SEQ: x_batch[1][0],
                ENCODER_MASK: x_batch[1][1],
                ENCODER_LABEL: x_batch[3],
                ENCODER_TYPES: params.src_encoders,
                FANOUTS: params.src_fanouts,
                FANOUTS_NAME: NODE_SRC,
            }

            enc.forward(context)
            output = context[EMBS_LIST]

    return output


def test_multitype_shared_feature_encoder(
    lp_mock_graph, prepare_local_twinbert_test_files
):
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    params, config = prepare_params(prepare_local_twinbert_test_files)
    output = run_multitype_feature_encoder(params, config, lp_mock_graph, shared=True)

    expected = np.array(
        [
            -0.210521,
            -0.251593,
            -0.308026,
            -0.189439,
            -0.401907,
            0.445968,
            -0.050585,
            0.173824,
            0.235175,
            -0.018321,
        ]
    )
    # just compare the first 10 values.
    npt.assert_allclose(output[0][0].detach().numpy()[0][0:10], expected, rtol=1e-4)


def test_multitype_feature_encoder(lp_mock_graph, prepare_local_twinbert_test_files):
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    params, config = prepare_params(prepare_local_twinbert_test_files)
    output = run_multitype_feature_encoder(params, config, lp_mock_graph, shared=False)

    expected = np.array(
        [
            -0.134458,
            0.17144,
            -0.217625,
            -0.089347,
            0.144573,
            0.090012,
            0.180561,
            0.045648,
            0.029409,
            -0.601354,
        ]
    )
    # just compare the first 10 values.
    npt.assert_allclose(output[0][0].detach().numpy()[0][0:10], expected, rtol=1e-4)


if __name__ == "__main__":
    sys.exit(
        pytest.main(
            [__file__, "--junitxml", os.environ["XML_OUTPUT_FILE"], *sys.argv[1:]]
        )
    )
