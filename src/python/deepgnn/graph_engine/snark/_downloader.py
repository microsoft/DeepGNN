# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Downoad graph data to a temporary folder."""
from typing import List, Union, Tuple
import tempfile

from deepgnn.graph_engine._base import get_fs
from fsspec.implementations.local import LocalFileSystem

from deepgnn.graph_engine.snark import meta as mt


class GraphPath:
    """A wrapper to handle both a temp dir and an existing path."""

    def __init__(self, path: Union[tempfile.TemporaryDirectory, str]):
        """Store a reference to the tempdir to prevent it from auto deletion."""
        self.path = path

    @property
    def name(self) -> str:
        """Return folder with graph binary data."""
        if isinstance(self.path, tempfile.TemporaryDirectory):
            return self.path.name
        return self.path

    def reset(self):
        """Remove up downloaded files."""
        if isinstance(self.path, tempfile.TemporaryDirectory):
            self.path.cleanup()


def download_graph_data(path: str, partitions: List[Tuple[str, int]]) -> GraphPath:
    """Download graph data to a temp folder if path is remote. If path is local, nothing is done."""
    fs, _ = get_fs(path)
    if isinstance(fs, LocalFileSystem):
        return GraphPath(path)
    data_path = tempfile.TemporaryDirectory(suffix="_snark")
    fs.get(mt._get_meta_path(path, fs.sep), mt._get_meta_path(data_path.name))
    graph_meta = mt.Meta(data_path.name)
    lpath = []
    for i in range(len(partitions)):
        partition_path, partition_index = partitions[i]
        fs, _ = get_fs(partition_path)
        if isinstance(fs, LocalFileSystem):
            continue

        lpath.append(mt._get_node_map_path(partition_path, partition_index, -1, fs.sep))
        lpath.append(
            mt._get_neighbors_index_path(partition_path, partition_index, -1, fs.sep)
        )
        lpath.append(
            mt._get_element_index_path(
                mt._Element.NODE, partition_path, partition_index, -1, fs.sep
            )
        )
        lpath.append(
            mt._get_element_index_path(
                mt._Element.EDGE, partition_path, partition_index, -1, fs.sep
            )
        )
        if graph_meta._node_feature_count > 0:
            lpath.append(
                mt._get_element_features_data_path(
                    mt._Element.NODE, partition_path, partition_index, -1, fs.sep
                )
            )
            lpath.append(
                mt._get_element_features_index_path(
                    mt._Element.NODE, partition_path, partition_index, -1, fs.sep
                )
            )
        if graph_meta._edge_feature_count > 0:
            lpath.append(
                mt._get_element_features_data_path(
                    mt._Element.EDGE, partition_path, partition_index, -1, fs.sep
                )
            )
            lpath.append(
                mt._get_element_features_index_path(
                    mt._Element.EDGE, partition_path, partition_index, -1, fs.sep
                )
            )

        # Use pattern matching to skip downloading alias tables if they were not created
        lpath.append(
            mt._get_element_alias_path(
                mt._Element.NODE, partition_path, -1, partition_index, fs.sep
            )
        )

        lpath.append(
            mt._get_element_alias_path(
                mt._Element.EDGE, partition_path, -1, partition_index, fs.sep
            )
        )

        partitions[i] = (data_path.name, partition_index)

    fs.get(lpath, data_path.name)
    return GraphPath(data_path)
