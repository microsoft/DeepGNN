# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import random
import json
import tempfile
import argparse
import os
from contextlib import closing
import multiprocessing as mp
from typing import List

import networkx as nx
import numpy as np
import numpy.testing as npt
import deepgnn.graph_engine.snark.convert as convert
import deepgnn.graph_engine.snark.decoders as decoders
import deepgnn.graph_engine.snark.server as server
import pytest

from deepgnn.graph_engine.backends.options import BackendOptions
from deepgnn.graph_engine.backends.snark.client import (
    SnarkLocalBackend,
    SnarkDistributedBackend,
)
from deepgnn.graph_engine.graph_dataset import BackendType
from deepgnn.graph_engine.snark.converter.options import DataConverterType


@pytest.fixture(scope="function")
def caveman_data():
    random.seed(246)
    g = nx.connected_caveman_graph(30, 12)
    nodes = []
    data = ""
    for node_id in g:
        # Set weights for neighbors
        nbs = {}
        for nb in nx.neighbors(g, node_id):
            nbs[nb] = 1.0

        node = {
            "node_weight": 1,
            "node_id": node_id,
            "node_type": 0,
            "uint64_feature": {},
            "float_feature": {"0": [node_id, random.random()]},
            "binary_feature": {},
            "edge": [
                {
                    "src_id": node_id,
                    "dst_id": nb,
                    "edge_type": 0,
                    "weight": 1.0,
                    "uint64_feature": {},
                    "float_feature": {},
                    "binary_feature": {},
                }
                for nb in nx.neighbors(g, node_id)
            ],
        }
        data += json.dumps(node) + "\n"
        nodes.append(node)

    working_dir = tempfile.TemporaryDirectory()

    raw_file = working_dir.name + "/data.json"
    with open(raw_file, "w+") as f:
        f.write(data)

    convert.MultiWorkersConverter(
        graph_path=raw_file,
        partition_count=2,
        output_dir=working_dir.name,
        decoder=decoders.JsonDecoder(),
    ).convert()

    yield working_dir.name


@pytest.fixture(scope="function")
def memory_graph(caveman_data):
    args = argparse.Namespace(
        data_dir=caveman_data,
        backend=BackendType.SNARK,
        local_ge=True,
        converter=DataConverterType.LOCAL,
        partitions=[0, 1],
        seed=23,
    )
    backend = SnarkLocalBackend(options=BackendOptions(args), is_leader=True)
    assert backend.graph is not None
    return backend.graph


@pytest.fixture(
    scope="function",
    params=[
        {
            # Standalone GE with explicit server
            "servers": ["localhost:12234"],
            "server_idx": 0,
            "client_rank": 0,
            "num_ge": None,
            "envs": {},
        },
        {
            # Standalone GE with defaults
            "servers": [],
            "server_idx": 0,
            "client_rank": 0,
            "num_ge": 1,
            "envs": {},
        },
        {
            # DDP single machine
            "servers": ["localhost:12234"],
            "server_idx": None,
            "client_rank": None,
            "num_ge": 0,
            "envs": {
                "LOCAL_RANK": "0",
                "LOCAL_WORLD_SIZE": "1",
                "WORLD_SIZE": "1",
                "RANK": "0",
            },
        },
        {
            # Horovod gloo
            "servers": None,
            "server_idx": None,
            "client_rank": None,
            "num_ge": 0,
            "envs": {
                "HOROVOD_LOCAL_RANK": "0",
                "HOROVOD_LOCAL_SIZE": "1",
                "HOROVOD_SIZE": "1",
                "HOROVOD_RANK": "0",
            },
        },
        {
            # Horovod openmpi
            "servers": [],
            "server_idx": None,
            "client_rank": None,
            "num_ge": 0,
            "envs": {
                "OMPI_COMM_WORLD_LOCAL_RANK": "0",
                "OMPI_COMM_WORLD_LOCAL_SIZE": "1",
                "OMPI_COMM_WORLD_RANK": "0",
                "OMPI_COMM_WORLD_SIZE": "1",
            },
        },
        {
            # TF_CONFIG
            "servers": ["localhost:12234"],
            "server_idx": None,
            "client_rank": None,
            "num_ge": 0,
            "envs": {
                "TF_CONFIG": '{"cluster": {"worker": ["localhost:2000", "localhost:2001"]}, "task": {"type": "worker", "index": 0}}'
            },
        },
        {
            # TF command line args
            "servers": None,
            "server_idx": None,
            "client_rank": None,
            "num_ge": 0,
            "envs": {},
            "task_index": 0,
            "worker_hosts": ["localhost:12234"],
        },
    ],
)
def distributed_graph(caveman_data, request):
    with tempfile.TemporaryDirectory(prefix="test_snark_client_") as workdir:
        args = argparse.Namespace(
            data_dir=caveman_data,
            model_dir=workdir,
            backend=BackendType.SNARK,
            local_ge=False,
            converter=DataConverterType.LOCAL,
            partitions=[0, 1],
            servers=request.param["servers"],
            server_idx=request.param["server_idx"],
            client_rank=request.param["client_rank"],
            seed=23,
            num_ge=request.param["num_ge"],
        )
        for k, v in request.param["envs"].items():
            os.environ[k] = v
        if "worker_hosts" in request.param:
            args.worker_hosts = request.param["worker_hosts"]
            args.task_index = request.param["task_index"]

        with closing(
            SnarkDistributedBackend(options=BackendOptions(args), is_leader=True)
        ) as backend:
            assert backend.graph is not None
            yield backend.graph

        for k, _ in request.param["envs"].items():
            del os.environ[k]


# Test requires to be forked to start/stop grpc properly across the whole module
@pytest.mark.forked
def test_snark_basic_distributed_backend(distributed_graph):
    values = distributed_graph.node_features(
        np.array([0, 13, 42], dtype=np.int64),
        np.array([[0, 2]], dtype=np.int32),
        np.float32,
    )

    assert values.shape == (3, 2)
    npt.assert_almost_equal(
        [[0, 0.9271], [13, 0.2157], [42, 0.4349]], values, decimal=4
    )


def test_snark_basic_local_backend(memory_graph):
    values = memory_graph.node_features(
        np.array([1], dtype=np.int64),
        np.array([[0, 2]], dtype=np.int32),
        np.float32,
    )

    assert values.shape == (1, 2)
    npt.assert_almost_equal([[1, 0.516677]], values, decimal=4)


@pytest.fixture(
    scope="function",
    params=[
        {
            "workers": [
                # Standalone GE with explicit server
                {
                    "servers": ["localhost:12235", "localhost:12236"],
                    "server_idx": 0,
                    "client_rank": 0,
                    "num_ge": 0,
                    "envs": {},
                    "partitions": [0],
                },
                {
                    "servers": ["localhost:12235", "localhost:12236"],
                    "server_idx": 1,
                    "client_rank": 1,
                    "num_ge": 0,
                    "envs": {},
                    "partitions": [1],
                },
            ],
            "expected_ge_count": 2,
        },
        {
            "workers": [
                # Autodetect servers
                {
                    "servers": [],
                    "server_idx": 0,
                    "client_rank": 0,
                    "num_ge": 0,
                    "envs": {},
                    "partitions": [0],
                },
                {
                    "servers": [],
                    "server_idx": 1,
                    "client_rank": 1,
                    "num_ge": 0,
                    "envs": {},
                    "partitions": [1],
                },
            ],
            # server_idx allows us to start multiple GEs on the same node.
            "expected_ge_count": 2,
        },
        {
            # DDP 2 workers, 1 GE
            "workers": [
                {
                    "servers": ["localhost:12237"],
                    "server_idx": None,
                    "client_rank": None,
                    "num_ge": 0,
                    "partitions": [0, 1],
                    "envs": {
                        "LOCAL_RANK": "0",
                        "LOCAL_WORLD_SIZE": "2",
                        "RANK": "0",
                        "WORLD_SIZE": "2",
                    },
                },
                {
                    "servers": ["localhost:12237"],
                    "server_idx": None,
                    "client_rank": None,
                    "num_ge": 0,
                    "partitions": [],
                    "envs": {
                        "LOCAL_RANK": "1",
                        "WORLD_SIZE": "2",
                        "RANK": "1",
                        "LOCAL_WORLD_SIZE": "2",
                    },
                },
            ],
            "expected_ge_count": 1,
        },
        {
            # MPI 1 node, 3 workers total. Can't autodetect multiple ips on a single test machine.
            "workers": [
                {
                    "servers": [],
                    "server_idx": None,
                    "client_rank": None,
                    "num_ge": 0,
                    "partitions": [0],
                    "envs": {
                        "OMPI_COMM_WORLD_LOCAL_RANK": "0",
                        "OMPI_COMM_WORLD_LOCAL_SIZE": "3",
                        "OMPI_COMM_WORLD_RANK": "0",
                        "OMPI_COMM_WORLD_SIZE": "3",
                    },
                },
                {
                    "servers": [],
                    "server_idx": None,
                    "client_rank": None,
                    "num_ge": 0,
                    "partitions": [],
                    "envs": {
                        "OMPI_COMM_WORLD_LOCAL_RANK": "1",
                        "OMPI_COMM_WORLD_LOCAL_SIZE": "3",
                        "OMPI_COMM_WORLD_RANK": "1",
                        "OMPI_COMM_WORLD_SIZE": "3",
                    },
                },
                {
                    "servers": [],
                    "server_idx": None,
                    "client_rank": None,
                    "num_ge": 0,
                    "partitions": [1],
                    "envs": {
                        "OMPI_COMM_WORLD_LOCAL_RANK": "2",
                        "OMPI_COMM_WORLD_LOCAL_SIZE": "3",
                        "OMPI_COMM_WORLD_RANK": "2",
                        "OMPI_COMM_WORLD_SIZE": "3",
                    },
                },
            ],
            "expected_ge_count": 1,
        },
        {
            # DDP 2 workers, 2 GE without autodetect
            "workers": [
                {
                    "servers": ["localhost:12238", "localhost:12239"],
                    "server_idx": None,
                    "client_rank": None,
                    "num_ge": 0,
                    "partitions": [0],
                    "envs": {
                        "LOCAL_RANK": "0",
                        "LOCAL_WORLD_SIZE": "1",
                        "WORLD_SIZE": "2",
                        "RANK": "0",
                    },
                },
                {
                    "servers": ["localhost:12238", "localhost:12239"],
                    "server_idx": None,
                    "client_rank": None,
                    "num_ge": 0,
                    "partitions": [1],
                    "envs": {
                        "LOCAL_RANK": "0",
                        "LOCAL_WORLD_SIZE": "1",
                        "RANK": "1",
                        "WORLD_SIZE": "2",
                    },
                },
            ],
            "expected_ge_count": 2,
        },
        {
            # TF_CONFIG 1 node, 3 workers total
            "workers": [
                {
                    "servers": [],
                    "server_idx": None,
                    "client_rank": None,
                    "num_ge": 0,
                    "partitions": [],
                    "envs": {
                        "TF_CONFIG": '{"cluster": {"worker": ["localhost:2000", "localhost:2001", "localhost:2002"]}, "task": {"type": "worker", "index": 1}}'
                    },
                },
                {
                    "servers": [],
                    "server_idx": None,
                    "client_rank": None,
                    "num_ge": 0,
                    "partitions": [],
                    "envs": {
                        "TF_CONFIG": '{"cluster": {"worker": ["localhost:2000", "localhost:2001", "localhost:2002"]}, "task": {"type": "worker", "index": 2}}'
                    },
                },
                {
                    "servers": [],
                    "server_idx": None,
                    "client_rank": None,
                    "num_ge": 0,
                    "partitions": [0, 1],
                    "envs": {
                        "TF_CONFIG": '{"cluster": {"worker": ["localhost:2000", "localhost:2001", "localhost:2002"]}, "task": {"type": "worker", "index": 0}}'
                    },
                },
            ],
            "expected_ge_count": 1,
        },
    ],
)
def distributed_graph_2_workers(caveman_data, request):
    expected_ge_count = request.param["expected_ge_count"]
    with tempfile.TemporaryDirectory(prefix="sync_dir") as sync_dir:
        with tempfile.TemporaryDirectory(prefix="test_snark_client_") as model_dir:

            def _make_backend(param, expected_ge_count):
                args = argparse.Namespace(
                    data_dir=caveman_data,
                    model_dir=model_dir,
                    backend=BackendType.SNARK,
                    local_ge=False,
                    converter=DataConverterType.SKIP,
                    partitions=param["partitions"],
                    sync_dir=sync_dir,
                    servers=param["servers"],
                    server_idx=param["server_idx"],
                    client_rank=param["client_rank"],
                    seed=23,
                    num_ge=param["num_ge"],
                    ge_start_timeout=10,
                )
                for k, v in param["envs"].items():
                    os.environ[k] = v
                if "worker_hosts" in param:
                    args.worker_hosts = param["worker_hosts"]
                    args.task_index = param["task_index"]
                options = BackendOptions(args)
                backend = SnarkDistributedBackend(
                    options=options, is_leader=param["server_idx"] == 0
                )
                assert len(options.servers) == expected_ge_count
                return backend

            event = mp.Event()

            def _start_backend(index):
                with closing(
                    _make_backend(request.param["workers"][index], expected_ge_count)
                ) as backend:
                    assert backend.graph is not None
                    event.wait()

            # We use the first worker to test and create client, the remaining workers will only initialize backend.
            processes: List[mp.Process] = []
            for index in range(1, len(request.param["workers"])):
                p = mp.Process(target=_start_backend, args=(index,))
                processes.append(p)
                p.start()

            with closing(
                _make_backend(request.param["workers"][0], expected_ge_count)
            ) as backend:
                assert backend.graph is not None
                yield backend.graph
                event.set()
            for p in processes:
                p.join()

            # Cleanup environment variables for next iteration
            for worker in request.param["workers"]:
                for k, _ in worker["envs"].items():
                    if k in os.environ:
                        del os.environ[k]


# Test requires to be forked to start/stop grpc properly across the whole module
@pytest.mark.forked
def test_snark_2_workers_1_client_distributed_backend(distributed_graph_2_workers):
    values = distributed_graph_2_workers.node_features(
        np.array([0, 13, 42], dtype=np.int64),
        np.array([[0, 2]], dtype=np.int32),
        np.float32,
    )

    assert values.shape == (3, 2)
    npt.assert_almost_equal(
        [[0, 0.9271], [13, 0.2157], [42, 0.4349]], values, decimal=4
    )


@pytest.mark.forked
def test_snark_backend_connect_to_existing_ge(caveman_data, request):
    host = "localhost:12240"
    srv = server.Server(caveman_data, [0, 1], hostname=host)
    with tempfile.TemporaryDirectory(prefix="test_snark_client_") as workdir:
        args = argparse.Namespace(
            data_dir=caveman_data,
            model_dir=workdir,
            backend=BackendType.SNARK,
            local_ge=False,
            converter=DataConverterType.SKIP,
            partitions=None,
            servers=[host],
            server_idx=None,
            client_rank=None,
            seed=23,
            num_ge=None,
            skip_ge_start=True,
        )
        with closing(
            SnarkDistributedBackend(options=BackendOptions(args), is_leader=True)
        ) as backend:
            assert backend.graph is not None
            values = backend.graph.node_features(
                np.array([0], dtype=np.int64),
                np.array([[0, 2]], dtype=np.int32),
                np.float32,
            )

            assert values.shape == (1, 2)
            npt.assert_almost_equal([[0, 0.9271]], values, decimal=4)
    srv.reset()
