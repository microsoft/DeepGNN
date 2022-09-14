# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Snark districuted client implementation."""
from typing import List, Dict
import tempfile

import deepgnn.graph_engine.snark.client as client
import deepgnn.graph_engine.snark.server as server
import deepgnn.graph_engine.snark.local as ge_snark
from ctypes import c_char_p
from deepgnn import get_logger
from deepgnn.graph_engine.snark.meta import _set_hadoop_classpath, _get_meta_path
from deepgnn.graph_engine.snark._lib import _get_c_lib


class Client(ge_snark.Client):
    """Distributed client."""

    def __init__(self, servers: List[str], ssl_cert: str = None):
        """Init snark client to wrapper around ctypes API of distributed graph."""
        self.logger = get_logger()
        self.logger.info(f"servers: {servers}. SSL: {ssl_cert}")
        self.graph = client.DistributedGraph(servers, ssl_cert)
        self.node_samplers: Dict[str, client.NodeSampler] = {}
        self.edge_samplers: Dict[str, client.EdgeSampler] = {}
        self.logger.info(
            f"Loaded distributed snark client. Node counts: {self.graph.meta.node_count_per_type}. Edge counts: {self.graph.meta.edge_count_per_type}"
        )


class Server:
    """Distributed server."""

    def __init__(
        self,
        hostname: str,
        data_path: str,
        index: int,
        total_shards: int,
        ssl_key: str = None,
        ssl_root: str = None,
        ssl_cert: str = None,
        storage_type: client.PartitionStorageType = client.PartitionStorageType.memory,
        config_path: str = "",
        stream: bool = False,
    ):
        """Init snark server."""
        path = data_path

        if (
            path.startswith("hdfs://")
            or path.startswith("adl://")
            or path.startswith("file:///")
        ):
            _set_hadoop_classpath(config_path)

            class _ErrCallback:  # Copied from client.py
                def __init__(self, method: str):
                    self.method = method

                # We have to use mypy ignore, to reuse this callable object across
                # all C function call because they have different signatures.
                def __call__(self, result, func, arguments):
                    if result != 0:
                        raise Exception(f"Failed to {self.method}")

            lib = _get_c_lib()
            lib.HDFSMoveMeta.errcheck = _ErrCallback("hdfs move meta")  # type: ignore

            hdfs_path = path
            temp_dir = tempfile.TemporaryDirectory()
            path = temp_dir.name
            lib.HDFSMoveMeta(
                c_char_p(bytes(_get_meta_path(hdfs_path), "utf-8")),
                c_char_p(bytes(_get_meta_path(path), "utf-8")),
                c_char_p(bytes(config_path, "utf-8")),
            )

        with open(_get_meta_path(path), "r") as meta:
            # TODO(alsamylk): expose graph metadata reader in snark.
            # Based on snark.client._read_meta() method
            skip_lines = 7
            for _ in range(skip_lines):
                meta.readline()
            partition_count = int(meta.readline())

        ssl_config = None
        if ssl_key is not None:
            ssl_config = {
                "ssl_key": ssl_key,
                "ssl_root": ssl_root,
                "ssl_cert": ssl_cert,
            }
        partitions = list(range(index, partition_count, total_shards))
        self.server = server.Server(
            data_path,
            partitions,
            hostname,
            ssl_config,  # type: ignore
            storage_type,
            config_path,
            stream,  # type: ignore
        )

    def reset(self):
        """Reset server."""
        if self.server is not None:
            self.server.reset()
