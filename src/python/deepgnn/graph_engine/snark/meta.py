# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Meta module provides functionality to work with binary graph files."""
import enum
import os
import json
import tempfile
from ctypes import c_char_p
from deepgnn.graph_engine.snark._lib import _get_c_lib
import platform

"""Version of binary files produced by converters to communicate breaking changes requiring regeneration of binary files."""
BINARY_DATA_VERSION = "v2"


# Use custom separators in case we want to download data from remote filesystems.
def _get_meta_path(path: str, sep=os.path.sep) -> str:
    return sep.join([path, "meta.json"])


def _get_node_map_path(
    path: str, partition: int, iteration: int, sep=os.path.sep
) -> str:
    suffix = "*" if iteration < 0 else str(iteration)
    return sep.join([path, "node_{}_{}.map".format(partition, suffix)])


class _Element(enum.Enum):
    NODE = "node"
    EDGE = "edge"


def _get_element_timestamps_path(
    element: _Element, path: str, partition: int, iteration: int, sep=os.path.sep
) -> str:
    suffix = "*" if iteration < 0 else str(iteration)
    return sep.join(
        [path, "{}_{}_{}.timestamp".format(element.value, partition, suffix)]
    )


def _get_element_index_path(
    element: _Element, path: str, partition: int, iteration: int, sep=os.path.sep
) -> str:
    suffix = "*" if iteration < 0 else str(iteration)
    return sep.join([path, "{}_{}_{}.index".format(element.value, partition, suffix)])


def _get_element_features_index_path(
    element: _Element, path: str, partition: int, iteration: int, sep=os.path.sep
) -> str:
    suffix = "*" if iteration < 0 else str(iteration)
    return sep.join(
        [path, "{}_features_{}_{}.index".format(element.value, partition, suffix)]
    )


def _get_element_sparse_features_index_path(
    element: _Element, path: str, partition: int, iteration: int, sep=os.path.sep
) -> str:
    suffix = "*" if iteration < 0 else str(iteration)
    return sep.join(
        [
            path,
            "{}_sparse_features_{}_{}.index".format(element.value, partition, suffix),
        ]
    )


def _get_element_features_data_path(
    element: _Element, path: str, partition: int, iteration: int, sep=os.path.sep
) -> str:
    suffix = "*" if iteration < 0 else str(iteration)
    return sep.join(
        [path, "{}_features_{}_{}.data".format(element.value, partition, suffix)]
    )


def _get_element_alias_path(
    element: _Element, path: str, tp: int, partition: int, sep=os.path.sep
) -> str:
    type_expr = "*" if tp < 0 else str(tp)
    return sep.join(
        [path, "{}_{}_{}.alias".format(element.value, type_expr, partition)]
    )


def _get_neighbors_index_path(
    path: str, partition: int, iteration: int, sep=os.path.sep
) -> str:
    suffix = "*" if iteration < 0 else str(iteration)
    return sep.join([path, "neighbors_{}_{}.index".format(partition, suffix)])


def _set_hadoop_classpath(config_dir):
    if platform.system() != "Linux":
        raise ValueError("HDFS only supported for linux!")
    if "CLASSPATH" in os.environ:
        return
    if config_dir.endswith(".xml"):
        config_dir = os.path.dirname(config_dir)
    hadoop_home = os.environ["HADOOP_HOME"]
    paths = set()
    for root, dirs, filenames in os.walk(hadoop_home):
        if any([filename.endswith(".jar") for filename in filenames]):
            if len(dirs):
                for dir in dirs:
                    paths.add(f"{root}/{dir}/*")
            else:
                paths.add(f"{root}/*")
    os.environ["CLASSPATH"] = f"{config_dir}:{':'.join(paths)}"


def download_meta(
    src_path: str,
    dest_path: str,
    config_path: str,
) -> str:
    """Get meta.json path based on the local or remote path."""
    path = src_path

    if (
        src_path.startswith("hdfs://")
        or src_path.startswith("adl://")
        or src_path.startswith("file:///")
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

        hdfs_path = src_path
        lib.HDFSMoveMeta(
            c_char_p(bytes(_get_meta_path(hdfs_path), "utf-8")),
            c_char_p(bytes(_get_meta_path(dest_path), "utf-8")),
            c_char_p(bytes(config_path, "utf-8")),
        )
        path = dest_path

    return _get_meta_path(path)


class Meta:
    """General information about the graph: number of node, edges, types, partititons, etc."""

    def __init__(self, path: str, config_path: str = ""):
        """
        Parse a metadata file generated by a converter.

        Args:
            path (str): location of graph binary files.
        """
        temp_dir = tempfile.TemporaryDirectory()
        temp_path = temp_dir.name
        meta_path = download_meta(path, temp_path, config_path)

        try:
            with open(meta_path, "r") as file:
                meta = json.load(file)  # type: ignore
        except FileNotFoundError:
            raise FileNotFoundError(
                "Failed to find meta.json file. Please use latest deepgnn package to convert data."
            )

        self.version = meta["binary_data_version"]
        if self.version[0] != "v":
            raise RuntimeError(
                "Meta file should contain binary_data_version, please regenerate binary data"
            )

        self.node_count = meta["node_count"]
        self.edge_count = meta["edge_count"]
        self.node_type_count = meta["node_type_count"]
        self.edge_type_count = meta["edge_type_count"]
        self._node_feature_count = meta["node_feature_count"]
        self._edge_feature_count = meta["edge_feature_count"]
        self.partition_count = len(meta["partitions"])
        self._node_weights = [0.0] * self.node_type_count
        self._edge_weights = [0.0] * self.edge_type_count
        for id in meta["partitions"]:
            for i in range(self.node_type_count):
                self._node_weights[i] += meta["partitions"][id]["node_weight"][i]
            for i in range(self.edge_type_count):
                self._edge_weights[i] += meta["partitions"][id]["edge_weight"][i]

        self.node_count_per_type = meta["node_count_per_type"]
        self.edge_count_per_type = meta["edge_count_per_type"]
        self.watermark = meta["watermark"]
