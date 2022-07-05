# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import json
import multiprocessing
import os
import tempfile
import http.server as http_server
from pathlib import Path
import platform
import sys

import numpy as np
import numpy.testing as npt
import pytest

import deepgnn.graph_engine.snark.client as client
from deepgnn.graph_engine.snark.decoders import JsonDecoder
import deepgnn.graph_engine.snark.convert as convert
import deepgnn.graph_engine.snark.server as server
import deepgnn.graph_engine.snark.dispatcher as dispatcher
import deepgnn.graph_engine.snark._lib as lib


def get_lib_name():
    lib_name = "libwrapper.so"
    if platform.system() == "Windows":
        lib_name = "wrapper.dll"
    return os.path.join(os.path.dirname(__file__), "..", lib_name)


def setup_module(_):
    lib._LIB_PATH = get_lib_name()


def small_graph_json(folder):
    data = open(os.path.join(folder, "graph.json"), "w+")
    graph = [
        {
            "node_id": 0,
            "node_type": 0,
            "node_weight": 1,
            "uint64_feature": {"0": [1, 2]},
            "edge": [
                {
                    "src_id": 0,
                    "dst_id": 1,
                    "edge_type": 0,
                    "weight": 1.0,
                    "uint64_feature": {},
                    "float_feature": {},
                    "binary_feature": {},
                }
            ],
        },
        {
            "node_id": 1,
            "node_type": 1,
            "node_weight": 1,
            "uint64_feature": {"0": [3, 4]},
            "float_feature": {},
            "binary_feature": {},
            "edge": [
                {
                    "src_id": 1,
                    "dst_id": 0,
                    "edge_type": 1,
                    "weight": 1,
                    "uint64_feature": {},
                    "float_feature": {},
                    "binary_feature": {},
                }
            ],
        },
    ]
    for el in graph:
        json.dump(el, data)
        data.write("\n")
    data.flush()

    meta = open(os.path.join(folder, "meta.txt"), "w+")
    meta.write(
        '{"node_type_num": 2, "edge_type_num": 2, \
        "node_uint64_feature_num": 1, "node_float_feature_num": 0, \
        "node_binary_feature_num": 0, "edge_uint64_feature_num": 0, \
        "edge_float_feature_num": 0, "edge_binary_feature_num": 0}'
    )
    meta.flush()
    data.close()
    meta.close()
    return data.name, meta.name


# We'll use this class for deterministic partitioning
class Counter:
    def __init__(self):
        self.count = 0

    def __call__(self, x):
        self.count += 1
        return self.count % 2


@pytest.fixture(
    scope="module", params=[(False, False), (False, True), (True, False), (True, True)]
)
def multi_partition_graph_data(request):
    output = tempfile.TemporaryDirectory()
    data_name, meta_name = small_graph_json(output.name)
    d = dispatcher.QueueDispatcher(
        Path(output.name), 2, meta_name, Counter(), JsonDecoder()
    )
    convert.MultiWorkersConverter(
        graph_path=data_name,
        meta_path=meta_name,
        partition_count=2,
        output_dir=output.name,
        decoder=JsonDecoder(),
        dispatcher=d,
        skip_node_sampler=request.param[0],
        skip_edge_sampler=request.param[1],
    ).convert()

    yield output.name


class _RunServer:
    def __init__(self, data_dir: str, start_event):  # type: ignore
        self.data_dir = data_dir
        self.start_event = start_event

    def __call__(self) -> None:
        data_dir = self.data_dir  # self is going to be hidden below.

        class CustomHandler(http_server.SimpleHTTPRequestHandler):
            def translate_path(self, path):  # type: ignore
                res = os.path.join(data_dir, *path.split("/")[2:])
                return res

        with http_server.HTTPServer(("localhost", 8181), CustomHandler) as httpd:
            self.start_event.set()
            httpd.serve_forever()


@pytest.fixture(scope="module")
def http_file_server(multi_partition_graph_data):
    start_event = multiprocessing.Event()
    server_process = multiprocessing.Process(
        target=_RunServer(multi_partition_graph_data, start_event)
    )

    server_process.start()
    start_event.wait()
    yield "http://localhost:8181/small"
    server_process.terminate()


platform_check = pytest.mark.skipif(
    sys.platform.startswith("darwin"), reason="tests are not stable"
)


@platform_check
def test_distributed_mode_feature(http_file_server):
    s = server.Server(http_file_server, [0, 1], "localhost:98765")
    c = client.DistributedGraph(["localhost:98765"])
    values = c.node_features(
        np.array([0, 1], dtype=np.int64),
        features=np.array([[0, 2]], dtype=np.int32),
        dtype=np.uint64,
    )
    npt.assert_equal(values, [[1, 2], [3, 4]])
    c.reset()
    s.reset()


@platform_check
def test_distributed_mode_sampling(http_file_server):
    s = server.Server(http_file_server, [0, 1], "localhost:98766")
    c = client.DistributedGraph(["localhost:98766"])
    sampler = client.NodeSampler(c, [0, 1])
    values = sampler.sample(5, seed=13)

    npt.assert_equal(values, [[0, 0, 0, 0, 1], [0, 0, 0, 0, 1]])
    c.reset()
    s.reset()


@platform_check
def test_memory_mode(http_file_server):
    c = client.MemoryGraph(http_file_server, partitions=[1])
    values = c.node_features(
        np.array([0, 1], dtype=np.int64),
        features=np.array([[0, 2]], dtype=np.int32),
        dtype=np.uint64,
    )
    npt.assert_equal(values, [[1, 2], [0, 0]])
    c.reset()


if __name__ == "__main__":
    sys.exit(
        pytest.main(
            [__file__, "--junitxml", os.environ["XML_OUTPUT_FILE"], *sys.argv[1:]]
        )
    )
