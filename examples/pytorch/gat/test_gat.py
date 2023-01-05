# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import platform
import sys
import pytest
import tempfile
import numpy as np
import torch
import argparse

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
from main import run_ray  # type: ignore
from deepgnn import get_logger


def setup_module(module):
    import deepgnn.graph_engine.snark._lib as lib

    lib_name = "libwrapper.so"
    if platform.system() == "Windows":
        lib_name = "wrapper.dll"

    os.environ[lib._SNARK_LIB_PATH_ENV_KEY] = os.path.join(
        os.path.dirname(__file__), "..", "..", "..", "src", "cc", "lib", lib_name
    )


@pytest.fixture(scope="module")
def train_graphsage_cora_ddp_trainer():
    model_dir = tempfile.TemporaryDirectory()
    working_dir = tempfile.TemporaryDirectory()
    Cora(working_dir.name)

    result = run_ray(
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
            "140",
            "--learning_rate",
            "0.005",
            "--num_epochs",
            "200",
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
            "1",
            "--algo",
            "supervised",
            "--neighbor_edge_types",
            "0",
            "--in_dim",
            "1433",
            "--head_num",
            "8,1",
            "--hidden_dim",
            "8",
            "--num_classes",
            "7",
            "--ffd_drop",
            "0.6",
            "--attn_drop",
            "0.6",
            "--sample_file",
            f"{working_dir.name}/train.nodes",
        ],
    )
    yield {
        "model_path": os.path.join(model_dir.name, "gnnmodel-196-000000.pt"),
        "data_dir": working_dir.name,
    }
    working_dir.cleanup()
    model_dir.cleanup()


def test_pytorch_gat_cora(train_graphsage_cora_ddp_trainer):
    qparam = GATQueryParameter(
        neighbor_edge_types=np.array([0], np.int32),
        feature_idx=0,
        feature_dim=1433,
        label_idx=1,
        label_dim=1,
    )
    model = GAT(
        in_dim=1433,
        head_num=[8, 1],
        hidden_dim=8,
        num_classes=7,
        ffd_drop=0.6,
        attn_drop=0.6,
        q_param=qparam,
    )

    model.load_state_dict(
        torch.load(train_graphsage_cora_ddp_trainer["model_path"])["state_dict"]
    )

    args = argparse.Namespace(
        data_dir=train_graphsage_cora_ddp_trainer["data_dir"],
        backend=BackendType.SNARK,
        graph_type=GraphType.LOCAL,
        converter=DataConverterType.SKIP,
        partitions=[0],
    )

    backend = create_backend(BackendOptions(args), is_leader=True)

    # evaluate
    def create_eval_dataset():
        ds = TorchDeepGNNDataset(
            sampler_class=FileNodeSampler,
            backend=backend,
            query_fn=model.q.query_training,
            prefetch_queue_size=1,
            prefetch_worker_size=1,
            batch_size=1000,
            sample_files=os.path.join(
                train_graphsage_cora_ddp_trainer["data_dir"], "test.nodes"
            ),
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
