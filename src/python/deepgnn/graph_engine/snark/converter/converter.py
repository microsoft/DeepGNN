# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Set of writers to convert graph from linar format to binary."""
import ctypes
import os
import tempfile
import struct
import sys
import typing

import numpy as np

import deepgnn.graph_engine.snark.alias as alias
import deepgnn.graph_engine.snark.meta as meta
from deepgnn.graph_engine._base import get_fs
from deepgnn.graph_engine.snark.meta import _Element


class NodeWriter:
    """
    NodeWriter is an entry point for conversion from euler json files to DeepGNN files.
    Every record is parsed one by one and passed to the relevant writers below. These writers
    put data in files with the partition suffix: `node_{partition}...`. We reserve one more
    suffix to follow after partition to split large files into smaller ones, so files will have names
    like this: `node_{partition}_{iteration}...`.
    """

    def __init__(self, folder: str, partition: int):
        """Initialize writer and create binary files.
        Args:
            folder (str): location to write output
            partition (int): first part of the suffix to identify files from this writer.
        """
        self.fs, _ = get_fs(folder)
        self.folder = folder
        self.partition = partition
        self.iteration = 0
        self.nm = self.fs.open(
            meta._get_node_map_path(folder, partition, self.iteration), "wb"
        )
        self.count = 0
        self.feature_writer = NodeFeatureWriter(folder, partition)

    def add(self, node_id, node_type, features):
        """Write node features and data about edges from this node to binary.
        Args:
            node (typing.Any): dictionary with information about
        """
        self.nm.write(
            struct.pack(
                "=QQi",
                node_id,
                self.count,
                node_type,
            )
        )
        self.feature_writer.add(features)
        self.count += 1

    def close(self):
        """Close output binary files."""
        self.nm.close()
        self.feature_writer.close()


class NodeFeatureWriter:
    """NodeFeatureWriter records node feature data in binary format."""

    def __init__(self, folder: str, partition: int):
        """Construct writer.
        Args:
            folder (str): path to output save output
            partition (int): first part of the suffix for binary files
        """
        self.fs, _ = get_fs(folder)
        self.folder = folder
        self.partition = partition
        self.iteration = 0
        self.ni = self.fs.open(
            meta._get_element_index_path(
                _Element.NODE, folder, partition, self.iteration
            ),
            "wb",
        )
        self.nfi = self.fs.open(
            meta._get_element_features_index_path(
                _Element.NODE, folder, partition, self.iteration
            ),
            "wb",
        )
        self.nfd = self.fs.open(
            meta._get_element_features_data_path(
                _Element.NODE, folder, partition, self.iteration
            ),
            "wb",
        )
        self.nfd_pos = self.nfd.tell()
        print("init", self.ndf_pos)

    def add(self, node: typing.Any):
        """Add node to binary output.
        Args:
            node (typing.Any): graph node with all node features and edges from it.
        """
        print(node)
        self.ni.write(ctypes.c_uint64(self.nfi.tell() // 8))  # type: ignore
        for k in convert_features(node):
            print(k, self.nfd_pos, ctypes.c_uint64(self.nfd_pos))
            # Fill the gaps between features
            self.nfi.write(ctypes.c_uint64(self.nfd_pos))  # type: ignore
            if k is not None:
                self.nfd_pos += self.nfd.write(k)

        print(self.nfd_pos, ctypes.c_uint64(self.nfd_pos))

    def close(self):
        """Close output binary files."""
        self.ni.write(ctypes.c_uint64(self.nfi.tell() // 8))
        self.ni.close()
        self.nfi.write(ctypes.c_uint64(self.nfd_pos))
        self.nfi.close()
        self.nfd.close()


class EdgeWriter:
    """EdgeWriter records information about neighbors and adds edge features to a EdgeFeatureWriter."""

    def __init__(self, folder: str, partition: int):
        """Initialize writer.
        Args:
            folder (str): path to save binary files
            partition (int): first part of a suffix in filename to identify partition
        """
        self.fs, _ = get_fs(folder)
        self.folder = folder
        self.partition = partition
        self.iteration = 0
        self.nbi = self.fs.open(
            meta._get_neighbors_index_path(folder, partition, self.iteration), "wb"
        )
        self.ei = self.fs.open(
            meta._get_element_index_path(
                _Element.EDGE, folder, partition, self.iteration
            ),
            "wb",
        )

        self.efi = self.fs.open(
            meta._get_element_features_index_path(
                _Element.EDGE, folder, partition, self.iteration
            ),
            "wb",
        )
        self.efi_pos = self.efi.tell()

        self.feature_writer = EdgeFeatureWriter(folder, partition, self.efi)

    def add(self, dst, tp, weight, features: typing.Any):
        """Append edges starting at node to the output.
        Args:
            node (typing.Any): node with edges data
        """
        # Record order is important for C++ reader: order fields by size for faster load.
        self.ei.write(
            struct.pack(
                "=QQif",
                dst,
                self.efi_pos // 8,
                tp,
                weight,
            )
        )
        self.efi_pos += self.feature_writer.add(features)

    def close(self):
        """Close output binary files."""
        self.nbi.write(
            ctypes.c_uint64(self.ei.tell() // (4 + 8 + 8 + 4))
        )  # type: ignore
        self.nbi.close()
        self.ei.write(
            struct.pack(
                "=QQif",
                0,
                self.efi.tell() // 8,
                0,
                -1,
            )
        )
        self.ei.close()
        self.efi.write(ctypes.c_uint64(self.feature_writer.tell()))  # type: ignore
        self.efi.close()
        self.feature_writer.close()


class EdgeFeatureWriter:
    """EdgeFeatureWriter records edge features one after each other in binary format."""

    def __init__(self, folder: str, partition: int, efi: typing.BinaryIO):
        """Create writer and open binary files for output.
        Args:
            folder (str): location to store binary files
            partition (int): suffix to identify binary files from this writer
            efi (typing.BinaryIO): feature index file descriptor
        """
        self.fs, _ = get_fs(folder)
        self.folder = folder
        self.partition = partition
        self.efi = efi
        self.iteration = 0
        self.efd = self.fs.open(
            meta._get_element_features_data_path(
                _Element.EDGE, folder, partition, self.iteration
            ),
            "wb",
        )
        self.efd_pos = self.efd.tell()

    def add(self, head: typing.Any):
        """Add edge features the binary output.
        Args:
            head (typing.Dict): collection of float/uint64/binary features.
        """
        count = 0
        for k in convert_features(head):
            count += self.efi.write(ctypes.c_uint64(self.efd_pos))  # type: ignore
            if k is not None:
                self.efd_pos += self.efd.write(k)

        return count

    def tell(self) -> int:
        """Tell start position for the next feature data.
        Returns:
            [int]: Last position in data file.
        """
        return self.efd_pos

    def close(self):
        """Close output files."""
        self.efd.close()


class NodeAliasWriter:
    """
    NodeAliasWriter creates alias tables for weighted node sampling.
    To avoid using lots of memory it utilizes temp files to store all nodes added to
    it and creates alias tables from them when the writer is closed.
    Each file has a fixed record format: node[8 bytes], alias node[8 bytes]
    and probability to pick node [4 bytes] over alias node.
    """

    def __init__(self, folder: str, partition: int, node_type_count: int):
        """Initialize alias tables.
        Args:
            folder (str): where to store files
            partition (int): suffix identifier for alias tables created with this writer.
            node_type_count (int): number of node types in the graph
        """
        self.fs, _ = get_fs(folder)
        self.folder = folder
        self.partition = partition
        self.node_type_count = node_type_count

        # generate temp alias file in local and then write final
        # results to destination. Hold the temp folder reference
        # to avoid deleting.
        self.meta_tmp_folder = tempfile.TemporaryDirectory()
        self.nodes = [
            open(
                "{}/tmp_alias_node_{}_{}.ids".format(
                    self.meta_tmp_folder.name, tp, partition
                ),
                "wb+",
            )
            for tp in range(node_type_count)
        ]
        self.weights = [
            open(
                "{}/tmp_alias_node_{}_{}.weights".format(
                    self.meta_tmp_folder.name, tp, partition
                ),
                "wb+",
            )
            for tp in range(node_type_count)
        ]

    def add(self, node_id, tp, weight):
        """Record node information.
        Args:
            node (typing.Any): Node with information about it's id, type and weight
        """
        self.nodes[tp].write(ctypes.c_uint64(node_id))
        self.weights[tp].write(ctypes.c_float(weight))

    def close(self):
        """Convert temporary files to the final alias tables."""
        for tp in range(self.node_type_count):
            wts = np.fromfile(
                self.weights[tp],
                dtype=np.float32,
                offset=-self.weights[tp].tell(),
                count=-1,
            )
            ids = np.fromfile(
                self.nodes[tp], dtype=np.int64, offset=-self.nodes[tp].tell(), count=-1
            )
            a = alias.Vose(ids, wts)
            with self.fs.open(
                meta._get_element_alias_path(
                    _Element.NODE, self.folder, tp, self.partition
                ),
                "wb",
            ) as nw:
                for index in range(len(a.elements)):
                    left = a.elements[index]
                    right = a.elements[a.alias[index]] if a.prob[index] < 1.0 else 0
                    nw.write(struct.pack("=qqf", left, right, a.prob[index]))

        for tp in range(self.node_type_count):
            self.weights[tp].close()
            os.remove(self.weights[tp].name)
            self.nodes[tp].close()
            os.remove(self.nodes[tp].name)

        self.meta_tmp_folder.cleanup()


class EdgeAliasWriter:
    """
    EdgeAliasWriter creates alias tables for edges.
    These tables can be used for weighted edge sampling with the same pattern
    as NodeAliasWriter. Final files have a fixed record format with following contents:
    (edge source[8 bytes], alias edge_source[8_bytes],
     edge destination[8 bytes], alias edge destination[8 bytes],
     probability [4 bytes] to select edge over an alias element).
    """

    def __init__(self, folder: str, partition: int, edge_type_count: int):
        """Initialize alias tables.
        Args:
            folder (str): where to store files
            partition (int): suffix identifier for alias tables created with this writer.
            edge_type_count (int): number of edge types in the graph
        """
        self.fs, _ = get_fs(folder)
        self.folder = folder
        self.partition = partition
        self.edge_type_count = edge_type_count

        # generate temp alias file in local and then write final
        # results to destination. Hold the temp folder reference
        # to avoid deleting.
        self.meta_tmp_folder = tempfile.TemporaryDirectory()
        self.pairs = [
            open(
                "{}/tmp_alias_edge_{}_{}.ids".format(
                    self.meta_tmp_folder.name, tp, partition
                ),
                "wb+",
            )
            for tp in range(edge_type_count)
        ]
        self.weights = [
            open(
                "{}/tmp_alias_edge_{}_{}.weights".format(
                    self.meta_tmp_folder.name, tp, partition
                ),
                "wb+",
            )
            for tp in range(edge_type_count)
        ]

    def add(self, src, dst, tp, weight: typing.Dict):
        """Add edge to the alias tables.
        Args:
            edge (typing.Dict): Edge with information about it's source/destination ids, type and weight
        """
        self.pairs[tp].write(ctypes.c_uint64(src))  # type: ignore
        self.pairs[tp].write(ctypes.c_uint64(dst))  # type: ignore
        self.weights[tp].write(ctypes.c_float(weight))  # type: ignore

    def close(self):
        """Convert temporary files to the final alias tables."""
        for tp in range(self.edge_type_count):
            wts = np.fromfile(
                self.weights[tp], dtype=np.float32, offset=-self.weights[tp].tell()
            )
            ids = np.fromfile(
                self.pairs[tp], dtype=np.int64, offset=-self.pairs[tp].tell()
            ).reshape(-1, 2)
            a = alias.Vose(ids, wts)
            with self.fs.open(
                meta._get_element_alias_path(
                    _Element.EDGE, self.folder, tp, self.partition
                ),
                "wb",
            ) as nw:
                for index in range(len(a.elements)):
                    left = a.elements[index]
                    right = (
                        a.elements[a.alias[index]] if a.prob[index] < 1.0 else (0, 0)
                    )
                    nw.write(
                        struct.pack(
                            "=qqqqf",
                            left[0],
                            left[1],
                            right[0],
                            right[1],
                            a.prob[index],
                        )
                    )

        for tp in range(self.edge_type_count):
            self.weights[tp].close()
            os.remove(self.weights[tp].name)
            self.pairs[tp].close()
            os.remove(self.pairs[tp].name)

        self.meta_tmp_folder.cleanup()


feature_items = [
    ("float_feature", ctypes.c_float),
    ("double_feature", ctypes.c_double),
    ("uint64_feature", ctypes.c_uint64),
    ("int64_feature", ctypes.c_int64),
    ("uint32_feature", ctypes.c_uint32),
    ("int32_feature", ctypes.c_int32),
    ("uint16_feature", ctypes.c_uint16),
    ("int16_feature", ctypes.c_int16),
    ("uint8_feature", ctypes.c_uint8),
    ("int8_feature", ctypes.c_int8),
]

def convert_features(features: list):
    """Convert the node's feature into bytes sorted by index."""
    # Use a single container for all node features to make sure there are unique feature_ids
    # and gaps between feature ids are processed correctly.
    output = []  # type: ignore
    for feature in features:
        if feature is None:
            output.append(feature)
        elif isinstance(feature, tuple):
            coordinates, values = feature
            assert (
                coordinates.shape[0] == len(values)
                if len(values) > 1
                else len(coordinates.shape) == 1
                or coordinates.shape[0]
                == 1  # relax input requirements for single values, both [[a,b]] and [a,b] are ok.
            ), f"Coordinates {coordinates} and values {values} dimensions don't match"

            coordinates_meta = bytes(
                ctypes.c_uint32(coordinates.shape[-1] if coordinates.ndim > 1 else 1)
            )

            if values.dtype == np.float16:
                values_buf = np.array(values, dtype=np.float16).tobytes()
            else:
                values_buf = (np.ctypeslib.as_ctypes_type(values.dtype) * len(values))()
                values_buf[:] = values

            # For matrices the number of values might be different than number of coordinates
            # Pack data in the following format: number of coordinates as uint32, then coordinates, and actual values in the end
            output.append(
                bytes(ctypes.c_uint32(coordinates.size))
                + coordinates_meta
                + bytes(coordinates.data)
                + values_buf
            )
        elif isinstance(feature, np.ndarray):
            output.append(feature.tobytes())
        else:
            output.append(bytes(feature, "utf-8"))

    return output
