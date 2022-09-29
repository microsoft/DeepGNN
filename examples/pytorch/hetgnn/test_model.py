# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
import numpy as np
import numpy.testing as npt
import os
import sys
import pytest
import random
import tempfile
import time

import torch
from torch.utils.data import IterableDataset
import ray
from ray.train.torch import TorchTrainer
from ray.air.config import ScalingConfig

from deepgnn import get_logger
from deepgnn.graph_engine.data.citation import Cora
from deepgnn.pytorch.training.args import get_args

from args import init_args  # type: ignore
from model import HetGnnModel  # type: ignore
from main import train_func  # type: ignore
#from sampler import HetGnnDataSampler  # type: ignore
#from conftest import (  # noqa: F401
#    load_data,
#    prepare_local_test_files,
#    init_het_input_data,
#)  # type: ignore
#import evaluation  # type: ignore
#import conftest  # type: ignore


logger = get_logger()


def get_train_args(data_dir, model_dir):
    args = [
            "--data_dir=" + data_dir,
            "--neighbor_count=10",
            "--model_dir=" + model_dir,
            "--save_path=" + model_dir,
            "--num_epochs=2",
            "--batch_size=128",
            "--walk_length=5",
            "--max_id=1024",
            "--node_type_count=1",
            "--neighbor_count=10",
            "--feature_idx=0",
            "--feature_dim=128",
            "--dim=128",
            "--sample_file="
            + os.path.join(data_dir, "train.nodes"),  # TODO probably need a join of all 3 files since hetgnn needs types>1
    ]
    return get_args(init_args, run_args=args)


@pytest.fixture(scope="session")
def train_academic_data():
    torch.manual_seed(0)
    np.random.seed(0)

    data_path = tempfile.TemporaryDirectory()
    model_path = tempfile.TemporaryDirectory()
    Cora(data_path.name)
    args = get_train_args(data_path.name, model_path.name)

    ray.init()
    trainer = TorchTrainer(
        train_func,
        train_loop_config=vars(args),
        scaling_config=ScalingConfig(num_workers=1, use_gpu=False),
    )
    result = trainer.fit()

    yield data_path.name, model_path.name, args
    data_path.cleanup()
    model_path.cleanup()


@pytest.fixture(scope="session")
def save_embedding(train_academic_data):
    data_dir, model_path, args = train_academic_data

    model = HetGnnModel(
        node_type_count=args.node_type_count,
        neighbor_count=args.neighbor_count,
        embed_d=args.dim,
        feature_type=np.float32,
        feature_idx=args.feature_idx,
        feature_dim=args.feature_dim,
    )
    model.load_state_dict(torch.load(os.path.join(model_path, "gnnmodel.pt")))
    model.eval()
    '''

    embed_file = open(os.path.join(model_path, "node_embedding.txt"), "w")

    batch_size = 200
    saving_dataset = MockHetGnnFileNodeLoader(
        graph=graph, batch_size=batch_size, model=model, sample_file=args.sample_file
    )

    data_loader = torch.utils.data.DataLoader(saving_dataset)
    for i, context in enumerate(data_loader):
        out_temp = model.get_embedding(context)
        out_temp = out_temp.data.cpu().numpy()
        inputs = context["inputs"].squeeze(0)
        for k in range(len(out_temp)):
            embed_file.write(
                str(inputs[k].numpy())
                + " "
                + " ".join([str(out_temp[k][x]) for x in range(len(out_temp[k]))])
                + "\n"
            )

    embed_file.close()

    return model_path

    '''

def test_link_prediction_on_het_gnn(
    save_embedding, init_het_input_data, tmpdir  # noqa: F811
):
    random.seed(0)
'''

    model_path = save_embedding
    input_data_map = init_het_input_data

    # do evaluation
    args = parse_testing_args([])
    train_num, test_num = conftest.a_a_collaborate_train_test(
        args, model_path, input_data_map, tmpdir
    )
    auc, f1 = evaluation.evaluate_link_prediction(args, train_num, test_num, tmpdir)

    assert auc > 0.6 and auc < 0.9
    assert f1 > 0.6 and f1 < 0.9


def test_classification_on_het_gnn(
    prepare_local_test_files, save_embedding, tmpdir  # noqa: F811
):
    random.seed(0)

    model_path = save_embedding

    # do evaluation
    args = parse_testing_args([])
    train_num, test_num, _ = conftest.a_class_cluster_feature_setting(
        args, model_path, tmpdir, prepare_local_test_files
    )
    macroF1, microF1 = evaluation.evaluate_node_classification(
        args, train_num, test_num, tmpdir
    )
    assert macroF1 > 0.9
    assert microF1 > 0.9


def test_academic_hetgnn_loss(mock_graph):
    torch.manual_seed(0)
    np.random.seed(0)

    node_type_count = 3
    neighbor_count = 10
    dim = 128
    feature_idx = 0
    feature_dim = 128
    learning_rate = 0.01
    batch_size = 128

    # train model
    model = HetGnnModel(
        node_type_count=node_type_count,
        neighbor_count=neighbor_count,
        embed_d=dim,
        feature_type=np.float32,
        feature_idx=feature_idx,
        feature_dim=feature_dim,
    )

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate,
        weight_decay=0,
    )

    backend_args = argparse.Namespace(
        data_dir="/mock/doesnt/need/physical/path",
        backend=BackendType.CUSTOM,
        graph_type=GraphType.LOCAL,
        converter=DataConverterType.SKIP,
        custom_backendclass=MockBackend,
    )
    backend = create_backend(BackendOptions(backend_args), is_leader=True)
    ds = TorchDeepGNNDataset(
        sampler_class=HetGnnDataSampler,
        backend=backend,
        query_fn=model.query,
        prefetch_queue_size=10,
        prefetch_worker_size=2,
        node_type_count=node_type_count,
        num_nodes=batch_size,
        batch_size=batch_size,
    )
    data_loader = torch.utils.data.DataLoader(ds, batch_size=None)

    it = iter(data_loader)
    optimizer.zero_grad()
    loss, _, _ = model(it.next())
    loss.backward()
    optimizer.step()
    print(loss.detach().numpy())
    npt.assert_allclose(loss.detach().numpy(), np.array([1.383309]), rtol=1e-5)

    try:
        next(it)
    except StopIteration:
        pass


def test_academic_hetgnn_model(mock_graph):
    torch.manual_seed(0)
    np.random.seed(0)

    node_type_count = 3
    neighbor_count = 10
    dim = 10
    feature_idx = 0
    feature_dim = 10
    batch_size = 128

    # train model
    model = HetGnnModel(
        node_type_count=node_type_count,
        neighbor_count=neighbor_count,
        embed_d=dim,
        feature_type=np.float32,
        feature_idx=feature_idx,
        feature_dim=feature_dim,
    )

    backend_args = argparse.Namespace(
        data_dir="/mock/doesnt/need/physical/path",
        backend=BackendType.CUSTOM,
        graph_type=GraphType.LOCAL,
        converter=DataConverterType.SKIP,
        custom_backendclass=MockBackend,
    )
    backend = create_backend(BackendOptions(backend_args), is_leader=True)
    ds = TorchDeepGNNDataset(
        sampler_class=HetGnnDataSampler,
        backend=backend,
        query_fn=model.query,
        prefetch_queue_size=10,
        prefetch_worker_size=2,
        num_nodes=batch_size,
        node_type_count=node_type_count,
        batch_size=batch_size,
    )
    data_loader = torch.utils.data.DataLoader(ds)

    it = iter(data_loader)
    index = 0
    context = it.next()
    feature_list = context["encoder"]
    for feature_index in range(len(feature_list)):
        if len(feature_list[feature_index]) > 0:
            index = feature_index
            break
    expected = [
        -0.041197,
        0.032675,
        0.104244,
        -0.125097,
        0.01668,
        -0.033342,
        -0.106841,
        0.015977,
        0.00044,
        0.000479,
    ]

    feature_list[index]["triple_index"] = index
    c_out_temp, _, _ = model.get_score(feature_list[index])
    npt.assert_allclose(
        c_out_temp[0].detach().numpy(), np.array(expected, dtype=np.float32), rtol=1e-3
    )

    try:
        next(it)
    except StopIteration:
        pass


def test_hetgnn_sampler_reset():
    torch.manual_seed(0)
    np.random.seed(0)

    node_type_count = 3
    neighbor_count = 10
    dim = 10
    feature_idx = 0
    feature_dim = 10
    batch_size = 128

    # train model
    model = HetGnnModel(
        node_type_count=node_type_count,
        neighbor_count=neighbor_count,
        embed_d=dim,
        feature_type=np.float32,
        feature_idx=feature_idx,
        feature_dim=feature_dim,
    )

    backend_args = argparse.Namespace(
        data_dir="/mock/doesnt/need/physical/path",
        backend=BackendType.CUSTOM,
        graph_type=GraphType.LOCAL,
        converter=DataConverterType.SKIP,
        custom_backendclass=MockBackend,
    )
    backend = create_backend(BackendOptions(backend_args), is_leader=True)
    ds = TorchDeepGNNDataset(
        sampler_class=HetGnnDataSampler,
        backend=backend,
        query_fn=model.query,
        prefetch_queue_size=10,
        prefetch_worker_size=2,
        num_nodes=batch_size,
        node_type_count=node_type_count,
        batch_size=batch_size,
    )

    count_expected = 0
    for _, _ in enumerate(ds):
        count_expected += 1

    count_actual = 0
    for _, _ in enumerate(ds):
        count_actual += 1

    assert count_expected > 0 and count_expected == count_actual

'''
if __name__ == "__main__":
    sys.exit(
        pytest.main(
            [__file__, "--junitxml", os.environ["XML_OUTPUT_FILE"], *sys.argv[1:]]
        )
    )
