# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import sys
import os
import tempfile
import pytest
import numpy as np
import numpy.testing as npt
import deepgnn.graph_engine.snark.client as client
import deepgnn.graph_engine.snark.server as server
from deepgnn.graph_engine.snark.client import PartitionStorageType
from deepgnn.graph_engine.data.cora import CoraFull


@pytest.fixture(scope="module")
def hdfs_data():
    working_dir = tempfile.TemporaryDirectory()
    data_dir = f"{working_dir.name}/cora"
    CoraFull(data_dir)
    yield data_dir
    working_dir.cleanup()


def test_hdfs_local(hdfs_data):
    repo_dir = os.environ["TEST_SRCDIR"].split("/sandbox")[0]
    if os.path.exists(f"{repo_dir}/bazel-snark"):
        hadoop_home = f"{repo_dir}/bazel-snark"
    elif os.path.exists(f"{repo_dir}/bazel-s"):
        hadoop_home = f"{repo_dir}/bazel-s"
    else:
        hadoop_home = f"{repo_dir.split('/bazel-out')[0]}"
    os.environ["HADOOP_HOME"] = f"{hadoop_home}/external/hadoop/"
    if "CLASSPATH" in os.environ:
        del os.environ["CLASSPATH"]
    data_path = "file://" + hdfs_data
    cl = client.MemoryGraph(data_path, [(data_path, 0)], config_path="", stream=True)

    v = cl.node_types(np.array([1, 2, 3], dtype=np.int64), default_type=-1)
    npt.assert_equal(v, np.array([0, 0, 0], dtype=np.int32))

    v = cl.node_features(
        np.array(range(12), dtype=np.int64),
        features=np.array([[1, 1]], dtype=np.int32),
        dtype=np.float32,
    )
    npt.assert_equal(
        v.flatten(),
        np.array([0.0, 0.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 0.0, 0.0, 4.0, 0.0]),
    )

    ns = client.NodeSampler(cl, [0, 2])
    v, t = ns.sample(size=5, seed=2)
    npt.assert_array_equal(v, [2203, 402, 1316, 2156, 41])
    npt.assert_array_equal(t, [0, 0, 0, 0, 0])


def test_hdfs_remote(hdfs_data):
    repo_dir = os.getcwd().split("/sandbox")[0]
    if os.path.exists(f"{repo_dir}/bazel-snark"):
        hadoop_home = f"{repo_dir}/bazel-snark"
    elif os.path.exists(f"{repo_dir}/bazel-s"):
        hadoop_home = f"{repo_dir}/bazel-s"
    else:
        hadoop_home = f"{repo_dir.split('/bazel-out')[0]}"
    os.environ["HADOOP_HOME"] = f"{hadoop_home}/external/hadoop/"
    if "CLASSPATH" in os.environ:
        del os.environ["CLASSPATH"]

    address = ["localhost:9999"]
    location = "file://" + hdfs_data
    s = server.Server(
        location, [(location, 0)], address[0], config_path="", stream=True
    )
    cl = client.DistributedGraph(address)

    v = cl.node_types(np.array([1, 2, 3], dtype=np.int64), default_type=-1)
    npt.assert_equal(v, np.array([0, 0, 0], dtype=np.int32))

    v = cl.node_features(
        np.array(range(12), dtype=np.int64),
        features=np.array([[1, 1]], dtype=np.int32),
        dtype=np.float32,
    )
    npt.assert_equal(
        v.flatten(),
        np.array([0.0, 0.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 0.0, 0.0, 4.0, 0.0]),
    )

    ns = client.NodeSampler(cl, [0, 2])
    v, t = ns.sample(size=5, seed=2)
    npt.assert_array_equal(v, [807, 72, 216, 1921, 1849])
    npt.assert_array_equal(t, [0, 0, 0, 0, 0])


def test_streaming_storage_args(hdfs_data):
    repo_dir = os.environ["TEST_SRCDIR"].split("/sandbox")[0]
    if os.path.exists(f"{repo_dir}/bazel-snark"):
        hadoop_home = f"{repo_dir}/bazel-snark"
    elif os.path.exists(f"{repo_dir}/bazel-s"):
        hadoop_home = f"{repo_dir}/bazel-s"
    else:
        hadoop_home = f"{repo_dir.split('/bazel-out')[0]}"
    os.environ["HADOOP_HOME"] = f"{hadoop_home}/external/hadoop/"
    if "CLASSPATH" in os.environ:
        del os.environ["CLASSPATH"]
    data_path = "file://" + hdfs_data

    with pytest.raises(
        ValueError,
        match="Use stream=False to download files first and use them from disk.",
    ):
        client.MemoryGraph(
            data_path,
            [(data_path, 0)],
            config_path="",
            stream=True,
            storage_type=PartitionStorageType.disk,
        )
    with pytest.raises(
        ValueError,
        match="Use stream=False to download files first and use them from disk.",
    ):
        server.Server(
            data_path,
            [(data_path, 0)],
            "localhost:0",
            config_path="",
            stream=True,
            storage_type=PartitionStorageType.disk,
        )


if __name__ == "__main__":
    sys.exit(
        pytest.main(
            [__file__, "--junitxml", os.environ["XML_OUTPUT_FILE"], "-s", *sys.argv[1:]]
        )
    )
