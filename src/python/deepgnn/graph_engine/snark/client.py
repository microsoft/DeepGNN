# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Clients to work with a graph in local and distributed mode."""
from datetime import datetime
import random
import os
import platform
import tempfile
from ctypes import (
    CFUNCTYPE,
    POINTER,
    Structure,
    byref,
    c_bool,
    c_char_p,
    c_float,
    c_int32,
    c_int64,
    c_uint64,
    c_uint8,
    c_size_t,
    c_uint32,
)
from typing import Any, List, Tuple, Union, Optional, Sequence, Dict
from enum import IntEnum

import json
import logging
import numpy as np
from tenacity import (
    Retrying,
    stop_after_attempt,
    wait_exponential,
    before_log,
    after_log,
)
from deepgnn import get_logger
from deepgnn.graph_engine.snark._lib import _get_c_lib
from deepgnn.graph_engine.snark._downloader import download_graph_data, GraphPath
from deepgnn.graph_engine.snark.meta import Meta


def _parse_grpc_options(options: List[Tuple[str, str]]) -> Dict:
    """Parse the grpc_options to use on client side retry calls."""
    for opt in options:
        if opt[0] == "grpc.service_config":
            try:
                config = json.loads(opt[1])
                max_attempts = int(
                    config["methodConfig"][0]["retryPolicy"]["maxAttempts"]
                )
                initial_backoff = datetime.strptime(
                    config["methodConfig"][0]["retryPolicy"]["initialBackoff"], "%Ss"
                ).second
                max_backoff = datetime.strptime(
                    config["methodConfig"][0]["retryPolicy"]["maxBackoff"], "%Ss"
                ).second
                backoff_multiplier = int(
                    config["methodConfig"][0]["retryPolicy"]["backoffMultiplier"]
                )
                return {
                    "max_attempts": max_attempts,
                    "initial_backoff": initial_backoff,
                    "max_backoff": max_backoff,
                    "backoff_multiplier": backoff_multiplier,
                }
            except Exception as err:
                get_logger().warning(f"Cannot parse the grpc_options: {err}")
                break

    # default options, delay in second
    get_logger().warning(
        "Setting default retry parameters to max_attempts=5, initial_backoff=2, max_backoff=10, backoff_multiplier=2"
    )
    return {
        "max_attempts": 5,
        "initial_backoff": 2,
        "max_backoff": 10,
        "backoff_multiplier": 2,
    }


class _DEEP_GRAPH(Structure):
    _fields_: List[Any] = []


class _ErrCallback:
    def __init__(self, method: str):
        self.method = method

    # We have to use mypy ignore, to reuse this callable object across
    # all C function call because they have different signatures.
    def __call__(self, result, func, arguments):
        if result != 0:
            raise Exception(f"Failed to {self.method}")


class PartitionStorageType(IntEnum):
    """Storage type for partition to use."""

    memory = 0
    disk = 1


# Define our own classes to copy data from C to Python runtime.
class _NeighborsCallback:
    def __init__(self, return_edge_created_ts):
        self.return_edge_created_ts = return_edge_created_ts
        self.node_ids = np.empty(0, dtype=np.int64)
        self.weights = np.empty(0, dtype=np.float32)
        self.edge_types = np.empty(0, dtype=np.int32)
        self.timestamps = np.empty(0, dtype=np.int64)

    def __call__(self, nodes, weights, types, timestamps, count):
        if count == 0:
            return

        self.node_ids = np.copy(np.ctypeslib.as_array(nodes, [count]))
        self.weights = np.copy(np.ctypeslib.as_array(weights, [count]))
        self.edge_types = np.copy(np.ctypeslib.as_array(types, [count]))
        if self.return_edge_created_ts:
            self.timestamps = np.copy(np.ctypeslib.as_array(timestamps, [count]))


_NEIGHBORS_CALLBACKFUNC = CFUNCTYPE(
    None,
    POINTER(c_int64),
    POINTER(c_float),
    POINTER(c_int32),
    POINTER(c_int64),
    c_size_t,
)


class _SparseFeatureCallback:
    def __init__(self, dtype, feature_len):
        self.indices = []
        self.feature_len = feature_len
        self.dimensions = np.empty(feature_len, dtype=np.int64)
        self.values = []
        self.dtype = dtype

    def __call__(self, indices, indices_len, values, values_len, dimensions):
        self.dimensions = np.copy(np.ctypeslib.as_array(dimensions, [self.feature_len]))
        for i in range(self.feature_len):
            if indices_len[i] == 0:
                self.indices.append(np.empty(0, dtype=np.int64))
                self.values.append(np.empty(0, dtype=self.dtype))
                continue

            # Increment original feature dimensions by 1, because we use node offset as
            # the first dimension and it is not present in binary format
            self.indices.append(
                np.copy(np.ctypeslib.as_array(indices[i], [indices_len[i]]))
                .astype(dtype=np.int64, copy=False)
                .reshape((-1, self.dimensions[i] + 1))
            )

            self.values.append(
                np.copy(np.ctypeslib.as_array(values[i], [values_len[i]])).view(
                    dtype=self.dtype
                )
            )


_SPARSE_FEATURE_CALLBACKFUNC = CFUNCTYPE(
    None,
    POINTER(POINTER(c_int64)),
    POINTER(c_size_t),
    POINTER(POINTER(c_uint8)),
    POINTER(c_size_t),
    POINTER(c_int64),
)


class _StringFeatureCallback:
    def __init__(self, dtype):
        self.values = np.empty(0, dtype=dtype)
        self.dtype = dtype

    def __call__(self, values_len, values):
        if values_len == 0:
            return
        self.values = np.copy(np.ctypeslib.as_array(values, [values_len])).view(
            dtype=self.dtype
        )


_STRING_FEATURE_CALLBACKFUNC = CFUNCTYPE(
    None,
    c_size_t,
    POINTER(c_uint8),
)


def _make_sorted_list(input: Union[int, List[int]]) -> List[int]:
    if isinstance(input, int):
        input = [input]
    return sorted(list(set(input)))


class MemoryGraph:
    """Graph stored fully in memory."""

    def __init__(
        self,
        meta_path: str,
        partitions: Optional[Sequence[Union[int, Tuple[str, int]]]] = None,
        storage_type: PartitionStorageType = PartitionStorageType.memory,
        config_path: str = "",
        stream: bool = False,
    ):
        """Load graph to memory.

        Args:
            meta_path: location of meta.json file with global graph information. If given hdfs:// or adl:// path
                use config_path and stream parameters, for additional configuration.
            partitions (Sequence[Union[int,Tuple[str, int]]]): Partition ids to load from meta_path folder.
                List can contain a tuple of path and partition id to load to override location of a binary file for given partition.
                For additional path configuration see `meta_path` parameter.
            storage_type (PartitionStorageType, default=memory): What type of feature / index storage to use in GE.
            config_path (str, optional): Path to folder with configuration files.
            stream (bool, default=False): If remote path is given: by default, download files first then load,
                if stream = True and libhdfs present, stream data directly to memory -- see docs/advanced/hdfs.md for setup and usage.
        """
        if storage_type == PartitionStorageType.disk and stream:
            raise ValueError(
                "Use stream=False to download files first and use them from disk."
            )

        if partitions is None:
            partitions = [(meta_path, 0)]
        partitions_with_path: List[Tuple[str, int]] = []
        for p in partitions:
            if isinstance(p, int):
                partitions_with_path.append((meta_path, p))
            elif len(p) == 2:
                assert isinstance(p[1], int) and isinstance(p[0], str)
                partitions_with_path.append((p[0], p[1]))
            else:
                assert False, f"Unrecognized type for partition {p}"

        self.seed = datetime.now()
        self.path = (
            GraphPath(meta_path)
            if stream
            else download_graph_data(meta_path, partitions_with_path)
        )
        self.meta = Meta(self.path.name, config_path)

        self.g_ = _DEEP_GRAPH()
        self.lib = _get_c_lib()

        self.lib.CreateLocalGraph.argtypes = [
            POINTER(_DEEP_GRAPH),
            c_char_p,
            c_size_t,
            POINTER(c_uint32),
            POINTER(c_char_p),
            c_int32,
            c_char_p,
        ]

        self.lib.CreateLocalGraph.errcheck = _ErrCallback(  # type: ignore
            "initialize graph"
        )

        PartitionArray = c_uint32 * len(partitions_with_path)
        partition_array = PartitionArray()
        for i in range(len(partitions_with_path)):
            partition_array[i] = c_uint32(partitions_with_path[i][1])

        LocationArray = c_char_p * len(partitions_with_path)
        location_array = LocationArray()
        for i in range(len(partitions_with_path)):
            location_array[i] = c_char_p(bytes(partitions_with_path[i][0], "utf-8"))

        self.lib.CreateLocalGraph(
            byref(self.g_),
            c_char_p(bytes(self.path.name, "utf-8")),
            c_size_t(len(partitions_with_path)),
            partition_array,
            location_array,
            c_int32(storage_type),
            c_char_p(bytes(config_path, "utf-8")),
        )
        self._describe_clib_functions()

        self._retryer = Retrying(stop=stop_after_attempt(0), reraise=True)  # noop retry

    def __del__(self):
        """Delete graph engine client."""
        self.reset()

    # Extract CDLL library functions descriptions in a separate method:
    # * describing C functions is not thread safe even if values are the same.
    # * assign argtypes and error callbacks once instead of inside relevant methods.
    def _describe_clib_functions(self):
        self.lib.GetNodeFeature.argtypes = [
            POINTER(_DEEP_GRAPH),
            POINTER(c_int64),
            c_size_t,
            POINTER(c_int64),
            POINTER(c_int32),
            c_size_t,
            POINTER(c_uint8),
            c_size_t,
        ]
        self.lib.GetNodeFeature.restype = c_int32
        self.lib.GetNodeFeature.errcheck = _ErrCallback(  # type: ignore
            "extract node features"
        )

        self.lib.GetEdgeFeature.argtypes = [
            POINTER(_DEEP_GRAPH),
            POINTER(c_int64),
            POINTER(c_int64),
            POINTER(c_int32),
            c_size_t,
            POINTER(c_int64),
            POINTER(c_int32),
            c_size_t,
            POINTER(c_uint8),
            c_size_t,
        ]
        self.lib.GetEdgeFeature.restype = c_int32
        self.lib.GetEdgeFeature.errcheck = _ErrCallback(  # type: ignore
            "extract edge features"
        )
        self.lib.GetNodeSparseFeature.argtypes = [
            POINTER(_DEEP_GRAPH),
            POINTER(c_int64),
            c_size_t,
            POINTER(c_int64),
            POINTER(c_int32),
            c_size_t,
            _SPARSE_FEATURE_CALLBACKFUNC,
        ]
        self.lib.GetNodeSparseFeature.restype = c_int32
        self.lib.GetNodeSparseFeature.errcheck = _ErrCallback(  # type: ignore
            "extract node sparse features"
        )

        self.lib.GetEdgeSparseFeature.argtypes = [
            POINTER(_DEEP_GRAPH),
            POINTER(c_int64),
            POINTER(c_int64),
            POINTER(c_int32),
            c_size_t,
            POINTER(c_int64),
            POINTER(c_int32),
            c_size_t,
            _SPARSE_FEATURE_CALLBACKFUNC,
        ]
        self.lib.GetEdgeSparseFeature.restype = c_int32
        self.lib.GetEdgeSparseFeature.errcheck = _ErrCallback(  # type: ignore
            "extract edge sparse features"
        )

        self.lib.GetNodeStringFeature.argtypes = [
            POINTER(_DEEP_GRAPH),
            POINTER(c_int64),
            c_size_t,
            POINTER(c_int64),
            POINTER(c_int32),
            c_size_t,
            POINTER(c_int64),
            _STRING_FEATURE_CALLBACKFUNC,
        ]
        self.lib.GetNodeStringFeature.restype = c_int32
        self.lib.GetNodeStringFeature.errcheck = _ErrCallback(  # type: ignore
            "extract node string features"
        )

        self.lib.GetEdgeStringFeature.argtypes = [
            POINTER(_DEEP_GRAPH),
            POINTER(c_int64),
            POINTER(c_int64),
            POINTER(c_int32),
            c_size_t,
            POINTER(c_int64),
            POINTER(c_int32),
            c_size_t,
            POINTER(c_int64),
            _STRING_FEATURE_CALLBACKFUNC,
        ]
        self.lib.GetEdgeStringFeature.restype = c_int32
        self.lib.GetEdgeStringFeature.errcheck = _ErrCallback(  # type: ignore
            "extract edge string features"
        )

        self.lib.NeighborCount.argtypes = [
            POINTER(_DEEP_GRAPH),
            POINTER(c_int64),
            c_size_t,
            POINTER(c_int64),
            POINTER(c_int32),
            c_size_t,
            POINTER(c_uint64),
        ]
        self.lib.NeighborCount.restype = c_int32
        self.lib.NeighborCount.errcheck = _ErrCallback(
            "get neighbor counts"
        )  # type: ignore

        self.lib.GetNeighbors.argtypes = [
            POINTER(_DEEP_GRAPH),
            c_bool,
            POINTER(c_int64),
            c_size_t,
            POINTER(c_int64),
            POINTER(c_int32),
            c_size_t,
            POINTER(c_uint64),
            _NEIGHBORS_CALLBACKFUNC,
        ]
        self.lib.GetNeighbors.restype = c_int32
        self.lib.GetNeighbors.errcheck = _ErrCallback("get neighbors")  # type: ignore

        self.lib.WeightedSampleNeighbor.argtypes = [
            POINTER(_DEEP_GRAPH),
            c_bool,
            c_int64,
            POINTER(c_int64),
            c_size_t,
            POINTER(c_int32),
            c_size_t,
            POINTER(c_int64),
            c_size_t,
            POINTER(c_int64),
            POINTER(c_int32),
            POINTER(c_float),
            POINTER(c_int64),
            c_int64,
            c_float,
            c_int32,
        ]
        self.lib.WeightedSampleNeighbor.restype = c_int32
        self.lib.WeightedSampleNeighbor.errcheck = _ErrCallback(  # type: ignore
            "extract sampler neighbors with weights"
        )

        self.lib.UniformSampleNeighbor.argtypes = [
            POINTER(_DEEP_GRAPH),
            c_bool,
            c_bool,
            c_int64,
            POINTER(c_int64),
            c_size_t,
            POINTER(c_int32),
            c_size_t,
            POINTER(c_int64),
            c_size_t,
            POINTER(c_int64),
            POINTER(c_int32),
            POINTER(c_int64),
            c_int64,
            c_int32,
        ]
        self.lib.UniformSampleNeighbor.restype = c_int32
        self.lib.UniformSampleNeighbor.errcheck = _ErrCallback(  # type: ignore
            "sample neighbors with uniform distribution"
        )

        self.lib.ResetGraph.argtypes = [POINTER(_DEEP_GRAPH)]
        self.lib.ResetGraph.restype = c_int32
        self.lib.ResetGraph.errcheck = _ErrCallback("reset graph")  # type: ignore

        self.lib.RandomWalk.argtypes = [
            POINTER(_DEEP_GRAPH),
            c_int64,
            c_float,
            c_float,
            c_int64,
            POINTER(c_int64),
            c_size_t,
            POINTER(c_int64),
            POINTER(c_int32),
            c_size_t,
            c_size_t,
            POINTER(c_int64),
        ]
        self.lib.RandomWalk.restype = c_int32
        self.lib.RandomWalk.errcheck = _ErrCallback("random walk")  # type: ignore

        self.lib.PPRSampleNeighbor.argtypes = [
            POINTER(_DEEP_GRAPH),
            POINTER(c_int64),
            c_size_t,
            POINTER(c_int64),
            POINTER(c_int32),
            c_size_t,
            c_size_t,
            c_float,
            c_float,
            c_int64,
            c_float,
            POINTER(c_int64),
            POINTER(c_float),
        ]
        self.lib.PPRSampleNeighbor.restype = c_int32
        self.lib.PPRSampleNeighbor.errcheck = _ErrCallback("ppr sample")  # type: ignore

        self.lib.LastNCreatedNeighbor.argtypes = [
            POINTER(_DEEP_GRAPH),
            c_bool,
            POINTER(c_int64),
            c_size_t,
            POINTER(c_int32),
            c_size_t,
            POINTER(c_int64),
            c_size_t,
            POINTER(c_int64),
            POINTER(c_int32),
            POINTER(c_float),
            POINTER(c_int64),
            c_int64,
            c_float,
            c_int32,
            c_int64,
        ]
        self.lib.LastNCreatedNeighbor.restype = c_int32
        self.lib.LastNCreatedNeighbor.errcheck = _ErrCallback(  # type: ignore
            "lastn sample"
        )

        self.lib.GetNodeType.argtypes = [
            POINTER(_DEEP_GRAPH),
            POINTER(c_int64),
            c_size_t,
            POINTER(c_int32),
            c_int32,
        ]
        self.lib.GetNodeType.restype = c_int32
        self.lib.GetNodeType.errcheck = _ErrCallback(  # type: ignore
            "extract node types"
        )

    def node_features(
        self,
        nodes: np.ndarray,
        features: np.ndarray,
        dtype: np.dtype,
        timestamps: np.ndarray = None,
    ) -> np.ndarray:
        """Retrieve node features.

        Args:
            nodes (np.array): list of nodes
            features (np.array): list of feature ids and sizes: [[feature_0, size_0], ..., [feature_n, size_n]]
            dtype (np.dtype): feature types to extract
            timestamps (np.array): list of timestamps corresponding to each node

        Returns:
            np.array: Features values ordered by node ids first and then by feature ids
        """
        nodes = np.array(nodes, dtype=np.int64)
        features = np.array(features, dtype=np.int32)
        assert features.shape[1] == 2

        result = np.zeros((len(nodes), features[:, 1].sum()), dtype=dtype)
        features_in_bytes = features.copy()
        features_in_bytes *= (1, result.itemsize)

        self._retryer(
            self.lib.GetNodeFeature,
            self.g_,
            nodes.ctypes.data_as(POINTER(c_int64)),
            c_size_t(len(nodes)),
            None
            if timestamps is None or len(timestamps) == 0
            else np.array(timestamps, dtype=np.int64).ctypes.data_as(POINTER(c_int64)),
            features_in_bytes.ctypes.data_as(POINTER(c_int32)),
            c_size_t(len(features)),
            result.ctypes.data_as(POINTER(c_uint8)),
            c_size_t(result.nbytes),
        )

        return result

    def node_sparse_features(
        self,
        nodes: np.ndarray,
        features: np.ndarray,
        dtype: np.dtype,
        timestamps: np.ndarray = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Retrieve node sparse features.

        Args:
            nodes (np.array): list of nodes
            features (np.array): list of feature ids
            dtype (np.dtype): feature types to extract
            timestamps (np.array): list of timestamps corresponding to each node

        Returns:
            List[np.ndarray]: List of coordinates for sparse features, with node index as first coordinate. Each list element represents coordinates of a corresponding feature.
            List[np.ndarray]: List of numpy arrays of sparse features values. List items ordered by feature ids passed as input.
            np.array: dimensions of returned sparse features.
        """
        nodes = np.array(nodes, dtype=np.int64)
        features = np.array(features, dtype=np.int32)
        assert len(features.shape) == 1

        py_cb = _SparseFeatureCallback(dtype, features.size)

        self._retryer(
            self.lib.GetNodeSparseFeature,
            self.g_,
            nodes.ctypes.data_as(POINTER(c_int64)),
            c_size_t(len(nodes)),
            None
            if timestamps is None or len(timestamps) == 0
            else np.array(timestamps, dtype=np.int64).ctypes.data_as(POINTER(c_int64)),
            features.ctypes.data_as(POINTER(c_int32)),
            c_size_t(len(features)),
            _SPARSE_FEATURE_CALLBACKFUNC(py_cb),
        )

        return py_cb.indices, py_cb.values, py_cb.dimensions

    def node_string_features(
        self,
        nodes: np.ndarray,
        features: np.ndarray,
        dtype: np.dtype,
        timestamps: np.ndarray = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Retrieve node string features, i.e. each feature has variable length.

        Args:
            nodes (np.array): list of nodes
            dtype (np.dtype): feature types to extract
            timestamps (np.array): list of timestamps corresponding to each node

        Returns:
            np.array: dimensions of returned features (#nodes, #features)
            np.array: feature values
        """
        nodes = np.array(nodes, dtype=np.int64)
        features = np.array(features, dtype=np.int32)
        assert len(features.shape) == 1
        dimensions = np.zeros((nodes.size, features.size), dtype=np.int64)
        py_cb = _StringFeatureCallback(dtype)

        self._retryer(
            self.lib.GetNodeStringFeature,
            self.g_,
            nodes.ctypes.data_as(POINTER(c_int64)),
            c_size_t(len(nodes)),
            None
            if timestamps is None or len(timestamps) == 0
            else np.array(timestamps, dtype=np.int64).ctypes.data_as(POINTER(c_int64)),
            features.ctypes.data_as(POINTER(c_int32)),
            c_size_t(len(features)),
            dimensions.ctypes.data_as(POINTER(c_int64)),
            _STRING_FEATURE_CALLBACKFUNC(py_cb),
        )

        return py_cb.values, dimensions // py_cb.values.itemsize

    def edge_features(
        self,
        edge_src: np.ndarray,
        edge_dst: np.ndarray,
        edge_tp: np.ndarray,
        features: np.ndarray,
        dtype: np.dtype,
        timestamps: np.ndarray = None,
    ) -> np.ndarray:
        """Retrieve edge features.

        Args:
            edges_src (np.array): array of edge source node_ids
            edges_dst (np.array): array of edge destination node_ids
            edges_tp (np.array): array of edge types
            features (np.array): list of feature ids and sizes: [[feature_0, size_0], ..., [feature_n, size_n]]
            dtype (np.dtype): list of corresponding feature dimensions
            timestamps (np.array): list of timestamps corresponding to each edge

        Returns:
            np.array: Features values ordered by edge ids first and then by feature ids
        """
        assert len(edge_src) == len(edge_dst)
        assert len(edge_src) == len(edge_tp)

        edge_src = np.array(edge_src, dtype=np.int64)
        edge_dst = np.array(edge_dst, dtype=np.int64)
        edge_tp = np.array(edge_tp, dtype=np.int32)
        features = np.array(features, dtype=np.int32)
        assert features.shape[1] == 2

        result = np.zeros((len(edge_src), features[:, 1].sum()), dtype=dtype)
        features_in_bytes = features.copy()
        features_in_bytes *= (1, result.itemsize)

        self._retryer(
            self.lib.GetEdgeFeature,
            self.g_,
            edge_src.ctypes.data_as(POINTER(c_int64)),
            edge_dst.ctypes.data_as(POINTER(c_int64)),
            edge_tp.ctypes.data_as(POINTER(c_int32)),
            c_size_t(len(edge_src)),
            None
            if timestamps is None or len(timestamps) == 0
            else np.array(timestamps, dtype=np.int64).ctypes.data_as(POINTER(c_int64)),
            features_in_bytes.ctypes.data_as(POINTER(c_int32)),
            c_size_t(len(features)),
            result.ctypes.data_as(POINTER(c_uint8)),
            c_size_t(result.nbytes),
        )

        return result

    def edge_sparse_features(
        self,
        edge_src: np.ndarray,
        edge_dst: np.ndarray,
        edge_tp: np.ndarray,
        features: np.ndarray,
        dtype: np.dtype,
        timestamps: np.ndarray = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Retrieve edge sparse features.

        Args:
            edges_src (np.array): array of edge source node_ids
            edges_dst (np.array): array of edge destination node_ids
            edges_tp (np.array): array of edge types
            features (np.array): list of feature ids
            dtype (np.dtype): Feature type as numpy array type.
            timestamps (np.array): list of timestamps corresponding to each edge

        Returns:
            List[np.ndarray]: List of coordinates for sparse features, with edge index as first coordinate. Each list element represents coordinates of a corresponding feature.
            List[np.ndarray]: List of numpy arrays of sparse features values. List items ordered by feature ids passed as input.
            np.array: dimensions of returned sparse features.
        """
        assert len(edge_src) == len(edge_dst)
        assert len(edge_src) == len(edge_tp)
        assert len(features.shape) == 1

        edge_src = np.array(edge_src, dtype=np.int64)
        edge_dst = np.array(edge_dst, dtype=np.int64)
        edge_tp = np.array(edge_tp, dtype=np.int32)
        features = np.array(features, dtype=np.int32)
        py_cb = _SparseFeatureCallback(dtype, features.size)

        self._retryer(
            self.lib.GetEdgeSparseFeature,
            self.g_,
            edge_src.ctypes.data_as(POINTER(c_int64)),
            edge_dst.ctypes.data_as(POINTER(c_int64)),
            edge_tp.ctypes.data_as(POINTER(c_int32)),
            c_size_t(len(edge_src)),
            None
            if timestamps is None or len(timestamps) == 0
            else np.array(timestamps, dtype=np.int64).ctypes.data_as(POINTER(c_int64)),
            features.ctypes.data_as(POINTER(c_int32)),
            c_size_t(len(features)),
            _SPARSE_FEATURE_CALLBACKFUNC(py_cb),
        )

        return py_cb.indices, py_cb.values, py_cb.dimensions

    def edge_string_features(
        self,
        edge_src: np.ndarray,
        edge_dst: np.ndarray,
        edge_tp: np.ndarray,
        features: np.ndarray,
        dtype: np.dtype,
        timestamps: np.ndarray = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Retrieve edge string features.

        Args:
            edges_src (np.array): array of edge source node_ids
            edges_dst (np.array): array of edge destination node_ids
            edges_tp (np.array): array of edge types
            features (np.array): list of feature ids
            dtype (np.dtype): Feature type as numpy array type.
            timestamps (np.array): list of timestamps corresponding to each edge

        Returns:
            np.array: dimensions of returned features (#edges, #features)
            np.array: feature values
        """
        assert len(edge_src) == len(edge_dst)
        assert len(edge_src) == len(edge_tp)

        edge_src = np.array(edge_src, dtype=np.int64)
        edge_dst = np.array(edge_dst, dtype=np.int64)
        edge_tp = np.array(edge_tp, dtype=np.int32)
        features = np.array(features, dtype=np.int32)
        assert len(features.shape) == 1

        dimensions = np.zeros((edge_src.size, features.size), dtype=np.int64)
        py_cb = _StringFeatureCallback(dtype)

        self._retryer(
            self.lib.GetEdgeStringFeature,
            self.g_,
            edge_src.ctypes.data_as(POINTER(c_int64)),
            edge_dst.ctypes.data_as(POINTER(c_int64)),
            edge_tp.ctypes.data_as(POINTER(c_int32)),
            c_size_t(len(edge_src)),
            None
            if timestamps is None or len(timestamps) == 0
            else np.array(timestamps, dtype=np.int64).ctypes.data_as(POINTER(c_int64)),
            features.ctypes.data_as(POINTER(c_int32)),
            c_size_t(len(features)),
            dimensions.ctypes.data_as(POINTER(c_int64)),
            _STRING_FEATURE_CALLBACKFUNC(py_cb),
        )

        return py_cb.values, dimensions // py_cb.values.itemsize

    def neighbor_counts(
        self,
        nodes: np.ndarray,
        edge_types: Union[List[int], int],
        timestamps: Union[List[int], np.ndarray] = None,
    ) -> np.ndarray:
        """Retrieve degree of node with satisfying edge types.

        Args:
            nodes -- array of nodes to select neighbors
            edge_types -- type of edges to use for selection.
            timestamps -- list of timestamps corresponding to each node

        Returns:
            np.ndarray: neighbor count
        """
        nodes = np.array(nodes, dtype=np.int64)
        edge_types = _make_sorted_list(edge_types)
        TypeArray = c_int32 * len(edge_types)
        etypes_arr = TypeArray(*edge_types)
        counts = np.empty(len(nodes), dtype=np.uint64)

        self._retryer(
            self.lib.NeighborCount,
            self.g_,
            nodes.ctypes.data_as(POINTER(c_int64)),
            nodes.size,
            None
            if timestamps is None or len(timestamps) == 0
            else np.array(timestamps, dtype=np.int64).ctypes.data_as(POINTER(c_int64)),
            etypes_arr,
            len(edge_types),
            counts.ctypes.data_as(POINTER(c_uint64)),
        )

        return counts

    def neighbors(
        self,
        nodes: np.ndarray,
        edge_types: Union[List[int], int],
        timestamps: Union[List[int], np.ndarray] = None,
        return_edge_created_ts: bool = False,
    ) -> Union[
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    ]:
        """
        Full list of node neighbors.

        nodes -- array of nodes to select neighbors
        edge_types -- type of edges to use for selection.
        timestamps -- list of timestamps corresponding to each node

        Returns a tuple of numpy arrays:
        -- neighbor ids per node, a one dimensional list of node ids
           concatenated in the same order as input nodes.
        -- weights for every neighbor, a one dimensional array
           of neighbor weights concatenated in the same order as input nodes.
        -- types of every neighbor.
        -- neighbor counts per node, with the shape [len(nodes)].
        -- timestamp when edge was created if return_edge_created_ts is set to True.
        """
        nodes = np.array(nodes, dtype=np.int64)
        edge_types = _make_sorted_list(edge_types)
        TypeArray = c_int32 * len(edge_types)
        etypes_arr = TypeArray(*edge_types)
        result_counts = np.empty(len(nodes), dtype=np.uint64)

        py_cb = _NeighborsCallback(return_edge_created_ts)

        self._retryer(
            self.lib.GetNeighbors,
            self.g_,
            c_bool(return_edge_created_ts),
            nodes.ctypes.data_as(POINTER(c_int64)),
            nodes.size,
            None
            if timestamps is None or len(timestamps) == 0
            else np.array(timestamps, dtype=np.int64).ctypes.data_as(POINTER(c_int64)),
            etypes_arr,
            len(edge_types),
            result_counts.ctypes.data_as(POINTER(c_uint64)),
            _NEIGHBORS_CALLBACKFUNC(py_cb),
        )

        if return_edge_created_ts:
            return (
                py_cb.node_ids,
                py_cb.weights,
                py_cb.edge_types,
                result_counts,
                py_cb.timestamps,
            )
        return py_cb.node_ids, py_cb.weights, py_cb.edge_types, result_counts

    def weighted_sample_neighbors(
        self,
        nodes: np.ndarray,
        edge_types: Union[List[int], int],
        count: int = 10,
        default_node: int = -1,
        default_weight: float = 0.0,
        default_edge_type: int = -1,
        seed: Optional[int] = None,
        timestamps: Optional[np.ndarray] = None,
        return_edge_created_ts: bool = False,
    ) -> Union[
        Tuple[np.ndarray, np.ndarray, np.ndarray],
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    ]:
        """Randomly sample neighbor nodes based on their weights(edge connecting 2 nodes).

        Args:
            nodes (np.array): list of nodes to sample neighbors from.
            edge_types (Union[List[int], int]): types of edges for neighbors selection.
            count (int, optional): Number of neighbors to sample. Defaults to 10.
            default_node (int, optional): Value to use if a node doesn't have neighbors. Defaults to -1.
            default_weight (float, optional): Weight to use for missing neighbors. Defaults to 0.0.
            seed (int, optional): Seed value for random samplers. Defaults to random.getrandbits(64).
            timestamps (Optional[np.ndarray], optional): Timestamps for nodes. Defaults to None.
            return_edge_created_ts (Optional[bool], optional): Return timestamps when edges connecting nodes were created,

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: a tuple of neighbor nodes, edge types and weights. If return_edge_created_ts specified,
            then this tuple will have a fourth element, list of timestamps when edges were created.
        """
        nodes = np.array(nodes, dtype=np.int64)
        edge_types = _make_sorted_list(edge_types)
        TypeArray = c_int32 * len(edge_types)
        etypes_arr = TypeArray(*edge_types)
        result_nodes = np.full((len(nodes), count), default_node, dtype=np.int64)
        result_types = np.full((len(nodes), count), default_edge_type, dtype=np.int32)
        result_weights = np.full((len(nodes), count), default_weight, dtype=np.float32)
        result_ts = np.empty(0)
        if return_edge_created_ts:
            assert (
                self.meta.watermark >= 0
            )  # return timestamps only for temporal graphs
            result_ts = np.full((len(nodes), count), -1, dtype=np.int64)

        self._retryer(
            self.lib.WeightedSampleNeighbor,
            self.g_,
            c_bool(return_edge_created_ts),
            c_int64(seed if seed is not None else random.getrandbits(64)),
            nodes.ctypes.data_as(POINTER(c_int64)),
            c_size_t(nodes.size),
            etypes_arr,
            c_size_t(len(edge_types)),
            None
            if timestamps is None or len(timestamps) == 0
            else np.array(timestamps, dtype=np.int64).ctypes.data_as(POINTER(c_int64)),
            c_size_t(count),
            result_nodes.ctypes.data_as(POINTER(c_int64)),
            result_types.ctypes.data_as(POINTER(c_int32)),
            result_weights.ctypes.data_as(POINTER(c_float)),
            result_ts.ctypes.data_as(POINTER(c_int64))
            if return_edge_created_ts
            else None,
            c_int64(default_node),
            c_float(default_weight),
            c_int32(default_edge_type),
        )

        if return_edge_created_ts:
            return result_nodes, result_weights, result_types, result_ts
        return result_nodes, result_weights, result_types

    def uniform_sample_neighbors(
        self,
        without_replacement: bool,
        nodes: np.ndarray,
        edge_types: Union[List[int], int],
        count: int = 10,
        default_node: int = -1,
        default_type: int = -1,
        seed: Optional[int] = None,
        timestamps: Optional[np.ndarray] = None,
        return_edge_created_ts: bool = False,
    ) -> Union[
        Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]
    ]:
        """Randomly sample neighbor nodes irrespectively of their weights.

        Args:
            without_replacement (bool): flag to replace selected neighbors from the population pool.
            nodes (np.array): list of nodes to sample neighbors from.
            edge_types (Union[List[int], int]): types of edges for neighbors selection.
            count (int, optional): Number of neighbors to sample. Defaults to 10.
            default_node (int, optional): Value to use if a node doesn't have neighbors. Defaults to -1.
            default_type (int, optional): Edge type to use for missing neighbors. Defaults to 0.
            seed (int, optional): Seed value for random samplers. Defaults to random.getrandbits(64).
            timestamps (Optional[np.ndarray], optional): Timestamps for each node. Defaults to None.
            return_edge_created_ts (Optional[bool], optional): Return timestamps when edges connecting nodes were created,

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: a tuple of neighbor nodes, edge types. If return_edge_created_ts specified,
            then this tuple will have a third element list of timestamps when edges were created.
        """
        nodes = np.array(nodes, dtype=np.int64)
        edge_types = _make_sorted_list(edge_types)

        TypeArray = c_int32 * len(edge_types)
        etypes_arr = TypeArray(*edge_types)
        result_nodes = np.full((len(nodes), count), default_node, dtype=np.int64)
        result_types = np.full((len(nodes), count), default_type, dtype=np.int32)
        result_ts = np.empty(0)
        if return_edge_created_ts:
            assert (
                self.meta.watermark >= 0
            )  # return timestamps only for temporal graphs
            result_ts = np.full((len(nodes), count), -1, dtype=np.int64)

        self._retryer(
            self.lib.UniformSampleNeighbor,
            self.g_,
            c_bool(without_replacement),
            c_bool(return_edge_created_ts),
            c_int64(seed if seed is not None else random.getrandbits(64)),
            nodes.ctypes.data_as(POINTER(c_int64)),
            c_size_t(nodes.size),
            etypes_arr,
            c_size_t(len(edge_types)),
            None
            if timestamps is None or len(timestamps) == 0
            else np.array(timestamps, dtype=np.int64).ctypes.data_as(POINTER(c_int64)),
            c_size_t(count),
            result_nodes.ctypes.data_as(POINTER(c_int64)),
            result_types.ctypes.data_as(POINTER(c_int32)),
            None
            if timestamps is None
            or len(timestamps) == 0
            or (not return_edge_created_ts)
            else result_ts.ctypes.data_as(POINTER(c_int64)),
            c_int64(default_node),
            c_int32(default_type),
        )

        if return_edge_created_ts:
            return result_nodes, result_types, result_ts
        return result_nodes, result_types

    def ppr_neighbors(
        self,
        nodes: np.ndarray,
        edge_types: Union[List[int], int],
        count: int = 10,
        alpha: float = 0.5,
        eps: float = 1e-4,
        default_node: int = -1,
        default_weight: float = 0.0,
        timestamps: Union[List[int], np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Personalized PageRank (PPR) sampling of neighbor nodes.

        Implementation is based on PPR-Go algorithm: https://github.com/TUM-DAML/pprgo_pytorch
        We don't return timestamps for this sampling method, because ppr-sampling involves multiple hops
        and timestamps for edges will be ambiguous.
        Args:
            nodes (np.array): list of nodes to sample neighbors from.
            edge_types (Union[List[int], int]): types of edges for neighbors selection.
            count (int, optional): Number of neighbors to sample. Defaults to 10.
            alpha (float, optional): PPR teleport probability. Defaults to 0.5.
            eps (float, optional): Stopping threshold for ACL's ApproximatePR. Defaults to 0.0001.
            default_node (int, optional): Value to use if a node doesn't have neighbors. Defaults to -1.
            default_weight (float, optional): Weight to use if a node doesn't have neighbors. Defaults to 0.

        Returns:
            Tuple[np.ndarray, np.ndarray]: a tuple of neighbor nodes, corresponding PR weights.
        """
        nodes = np.array(nodes, dtype=np.int64)
        edge_types = _make_sorted_list(edge_types)
        TypeArray = c_int32 * len(edge_types)
        etypes_arr = TypeArray(*edge_types)
        result_nodes = np.full((len(nodes), count), default_node, dtype=np.int64)
        result_weights = np.full((len(nodes), count), default_weight, dtype=np.float32)
        self.lib.PPRSampleNeighbor(
            self.g_,
            nodes.ctypes.data_as(POINTER(c_int64)),
            c_size_t(nodes.size),
            None
            if timestamps is None or len(timestamps) == 0
            else np.array(timestamps, dtype=np.int64).ctypes.data_as(POINTER(c_int64)),
            etypes_arr,
            c_size_t(len(edge_types)),
            c_size_t(count),
            c_float(alpha),
            c_float(eps),
            c_int64(default_node),
            c_float(default_weight),
            result_nodes.ctypes.data_as(POINTER(c_int64)),
            result_weights.ctypes.data_as(POINTER(c_float)),
        )

        return result_nodes, result_weights

    def lastn_neighbors(
        self,
        nodes: np.ndarray,
        edge_types: Union[np.ndarray, List[int], int],
        timestamps: np.ndarray,
        count: int = 10,
        default_node: int = -1,
        default_weight: float = 0.0,
        default_edge_type: int = -1,
        default_timestamp: int = -1,
        return_edge_created_ts: bool = False,
    ) -> Union[
        Tuple[np.ndarray, np.ndarray, np.ndarray],
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    ]:
        """Select last N neighbor nodes in temporal graphs.

        Args:
            nodes (ndarray): list of nodes to select neighbors from.
            edge_types (Union[np.ndarray, List[int], int]): types of edges for neighbors selection.
            timestamps (ndarray): Timestamps for nodes.
            count (int, optional): Number of neighbors to sample. Defaults to 10.
            default_node (int, optional): Value to use if a node doesn't have neighbors. Defaults to -1.
            default_weight (float, optional): Weight to use for missing neighbors. Defaults to 0.0.
            default_edge_type (int, optional): Edge type to use for missing neighbors. Defaults to -1.
            default_ts (int, optional): Timestamp to use for missing neighbors. Defaults to -1.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: a tuple of neighbor nodes, edge weights and types.
            If return_edge_created_ts argument is set to true, then the returned tuple contains an extra
            array with timestamps when corresponding edge was created.
        """
        assert self.meta.watermark >= 0
        nodes = np.array(nodes, dtype=np.int64)
        timestamps = np.array(timestamps, dtype=np.int64)
        assert len(nodes) == len(timestamps)
        edge_types = _make_sorted_list(edge_types)
        TypeArray = c_int32 * len(edge_types)
        etypes_arr = TypeArray(*edge_types)
        result_nodes = np.full((len(nodes), count), default_node, dtype=np.int64)
        result_types = np.full((len(nodes), count), default_edge_type, dtype=np.int32)
        result_weights = np.full((len(nodes), count), default_weight, dtype=np.float32)
        result_ts = np.empty(0)
        if return_edge_created_ts:
            result_ts = np.full((len(nodes), count), default_timestamp, dtype=np.int64)

        self._retryer(
            self.lib.LastNCreatedNeighbor,
            self.g_,
            c_bool(return_edge_created_ts),
            nodes.ctypes.data_as(POINTER(c_int64)),
            c_size_t(nodes.size),
            etypes_arr,
            c_size_t(len(edge_types)),
            timestamps.ctypes.data_as(POINTER(c_int64)),
            c_size_t(count),
            result_nodes.ctypes.data_as(POINTER(c_int64)),
            result_types.ctypes.data_as(POINTER(c_int32)),
            result_weights.ctypes.data_as(POINTER(c_float)),
            result_ts.ctypes.data_as(POINTER(c_int64))
            if return_edge_created_ts
            else None,
            c_int64(default_node),
            c_float(default_weight),
            c_int32(default_edge_type),
            c_int64(default_timestamp),
        )

        if return_edge_created_ts:
            return result_nodes, result_weights, result_types, result_ts
        return result_nodes, result_weights, result_types

    def reset(self):
        """Reset graph and unload it from memory."""
        self.lib.ResetGraph(self.g_)
        self.path.reset()

    def get_node_type_count(self, types: List[int]) -> int:
        """Return the number of nodes of specified types."""
        assert len(types) > 0
        result = 0
        for t in set(types):
            if t >= 0 and t < self.meta.node_type_count:
                result += self.meta.node_count_per_type[t]
        return result

    def get_edge_type_count(self, types: List[int]) -> int:
        """Return the number of edges of specified types."""
        assert len(types) > 0
        result = 0
        for t in set(types):
            if t >= 0 and t < self.meta.edge_type_count:
                result += self.meta.edge_count_per_type[t]
        return result

    def random_walk(
        self,
        node_ids: np.ndarray,
        edge_types: Union[List[int], int],
        walk_len: int,
        p: float,
        q: float,
        default_node: int = -1,
        seed: Optional[int] = None,
        timestamps: Optional[Union[List[int], np.ndarray]] = None,
    ) -> np.ndarray:
        """
        Sample nodes via random walk.

        node_ids: starting nodes
        edge_types (Union[List[int], int]): types of edges for neighbors selection.
        walk_len: number of steps to make
        p: return parameter, 1/p is unnormalized probability to return to a parent node
        q: in-out parameter, 1/q is unnormalized probability to select a neighbor not connected to a parent
        nodes connected to both a parent and a current node will be selected with unnormalized probability 1.
        default_node: default node id if a neighbor cannot be retrieved
        seed: seed to feed random generator
        timestamps: timestamps corresponding to starting nodes
        Returns starting and neighbor nodes visited during the walk
        """
        node_ids = np.array(node_ids, dtype=np.int64)
        edge_types = _make_sorted_list(edge_types)

        TypeArray = c_int32 * len(edge_types)
        etypes_arr = TypeArray(*edge_types)
        result_nodes = np.empty((len(node_ids), walk_len + 1), dtype=np.int64)

        self._retryer(
            self.lib.RandomWalk,
            self.g_,
            c_int64(random.getrandbits(64) if seed is None else seed),
            c_float(p),
            c_float(q),
            c_int64(default_node),
            node_ids.ctypes.data_as(POINTER(c_int64)),
            c_size_t(node_ids.size),
            None
            if timestamps is None or len(timestamps) == 0
            else np.array(timestamps, dtype=np.int64).ctypes.data_as(POINTER(c_int64)),
            etypes_arr,
            c_size_t(len(edge_types)),
            c_size_t(walk_len),
            result_nodes.ctypes.data_as(POINTER(c_int64)),
        )

        return result_nodes

    def node_types(
        self,
        nodes: np.ndarray,
        default_type: int,
    ) -> np.ndarray:
        """Retrieve node types.

        Args:
            nodes (np.array): list of nodes
            default_type (int): default value to use for nodes not found in the graph

        Returns:
            np.array: types of the input nodes
        """
        nodes = np.array(nodes, dtype=np.int64)
        result = np.empty(len(nodes), dtype=np.int32)

        self._retryer(
            self.lib.GetNodeType,
            self.g_,
            nodes.ctypes.data_as(POINTER(c_int64)),
            c_size_t(len(nodes)),
            result.ctypes.data_as(POINTER(c_int32)),
            c_int32(default_type),
        )

        return result


class DistributedGraph(MemoryGraph):
    """Client for distributed graph."""

    def __init__(
        self,
        servers: List[str],
        ssl_cert: Optional[str] = None,
        num_threads: Optional[int] = None,
        num_cq_per_thread: Optional[int] = None,
        grpc_options: Optional[List[Tuple[str, str]]] = None,
    ):
        """Create a client to work with a graph in a distributed mode.

        Args:
            servers (List[str]): List of server hostnames to connect to.
            ssl_cert (str, optional): Certificates to use for connection if needed. Defaults to None.
            num_threads(int, optional): Number of threads to used for processing replies.
            num_cq_per_thread(int, optional): Number of completion queues to use per thread.
            grpc_options(List[Tuple(str, str)], optional): additional arguments to configure grpc client.
        """
        assert len(servers) > 0
        self.g_ = _DEEP_GRAPH()
        self.lib = _get_c_lib()

        self.lib.CreateRemoteClient.argtypes = [
            POINTER(_DEEP_GRAPH),
            c_char_p,
            POINTER(c_char_p),
            c_size_t,
            c_char_p,
            c_size_t,
            c_size_t,
            c_size_t,
            POINTER(c_char_p),
            POINTER(c_char_p),
        ]

        ServersArray = c_char_p * len(servers)
        pointers = ServersArray()
        for i, path in enumerate(servers):
            pointers[i] = c_char_p(bytes(path, "utf-8"))

        self.lib.CreateRemoteClient.errcheck = _ErrCallback(  # type: ignore
            "initialize remote client"
        )
        if num_threads is None:
            if platform.system() == "Linux":
                num_threads = max(len(os.sched_getaffinity(0)), 1)  # type: ignore
            else:
                num_threads = 1
        if num_cq_per_thread is None:
            num_cq_per_thread = 1

        if grpc_options is None:
            grpc_options = []
        GRPCOptions = c_char_p * len(grpc_options)
        grpc_options_keys_pointers = GRPCOptions()
        grpc_options_values_pointers = GRPCOptions()
        for i, kv in enumerate(grpc_options):
            grpc_options_keys_pointers[i] = c_char_p(bytes(str(kv[0]), "utf-8"))
            grpc_options_values_pointers[i] = c_char_p(bytes(str(kv[1]), "utf-8"))

        with tempfile.TemporaryDirectory() as meta_dir:
            self.lib.CreateRemoteClient(
                byref(self.g_),
                c_char_p(bytes(str(meta_dir), "utf-8")),
                pointers,
                c_size_t(len(pointers)),
                c_char_p(bytes(ssl_cert, "utf-8")) if ssl_cert is not None else None,
                c_size_t(num_threads),
                c_size_t(num_cq_per_thread),
                c_size_t(len(grpc_options)),
                grpc_options_keys_pointers,
                grpc_options_values_pointers,
            )
            self.meta = Meta(meta_dir)
            # Keep an empty object to avoid ifs
            self.path = GraphPath("")

        # Using grpc options on DistributedGraph API calls
        retry_ops = _parse_grpc_options(grpc_options)
        self._retryer = Retrying(
            stop=stop_after_attempt(retry_ops["max_attempts"]),
            wait=wait_exponential(
                multiplier=retry_ops["backoff_multiplier"],
                min=retry_ops["initial_backoff"],
                max=retry_ops["max_attempts"],
            ),
            before=before_log(get_logger(), logging.DEBUG),
            after=after_log(get_logger(), logging.WARNING),
            reraise=True,
        )

        super()._describe_clib_functions()


class NodeSampler:
    """Sampler to fetch nodes from a graph."""

    def __init__(self, g: MemoryGraph, types: Union[List, int], kind: str = "weighted"):
        """Create sampler from the graph.

        Args:
            g (MemoryGraph): graph to use for sampling.
            types (Union[List, int]): node types to sample.
            kind (str, optional): sampling strategy. Defaults to "weighted".
        """
        if isinstance(types, int) and types == -1:
            types = []
            for tp in range(g.meta.node_type_count):
                types.append(tp)

        if isinstance(types, int):
            types = [types]

        types = _make_sorted_list(types)
        assert isinstance(types, List)

        types = list(
            filter(
                lambda tp: tp >= 0
                and tp < g.meta.node_type_count
                and g.meta.node_count_per_type[tp] > 0,
                types,
            )
        )

        assert (
            len(types) > 0
        ), "Sampler doesn't have any nodes to select from the graph. Check graph.node_count_by_type to inspect node counts"

        self.types = types
        self.ns_ = _DEEP_GRAPH()
        self.graph = g
        self.lib = g.lib

        sampler_func = {
            "weighted": self.lib.CreateWeightedNodeSampler,
            "uniform": self.lib.CreateUniformNodeSampler,
            "withoutreplacement": self.lib.CreateUniformNodeSamplerWithoutReplacement,
        }

        sampler_func[kind].argtypes = [
            POINTER(_DEEP_GRAPH),
            POINTER(_DEEP_GRAPH),
            c_size_t,
            POINTER(c_int32),
        ]
        sampler_func[kind].restype = c_int32
        sampler_func[kind].errcheck = _ErrCallback(  # type: ignore
            "create node sampler"
        )

        TypeArray = c_int32 * len(types)
        types_array = TypeArray(*types)
        sampler_func[kind](
            byref(self.graph.g_), byref(self.ns_), len(types), types_array
        )
        self._describe_clib_functions()

    def _describe_clib_functions(self):
        self.lib.SampleNodes.argtypes = [
            POINTER(_DEEP_GRAPH),
            c_int64,
            c_size_t,
            POINTER(c_int64),
            POINTER(c_int32),
        ]
        self.lib.SampleNodes.restype = c_int32
        self.lib.SampleNodes.errcheck = _ErrCallback("sample nodes")  # type: ignore

        self.lib.ResetSampler.argtypes = [POINTER(_DEEP_GRAPH)]
        self.lib.ResetSampler.restype = c_int32
        self.lib.ResetSampler.errcheck = _ErrCallback(  # type: ignore
            "reset node sampler"
        )

    def __del__(self):
        """Delete node sampler."""
        self.reset()

    def sample(
        self, size: int, seed: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Perform sampling.

        Args:
            size (int): number of nodes to sample.
            seed (int, optional): seed value to use for sampling. Defaults to random.getrandbits(64).

        Returns:
            Tuple[np.ndarray, np.ndarray]: a tuple of nodes and their corresponding types.
        """
        result_nodes = np.empty(size, dtype=np.int64)
        result_types = np.empty(size, dtype=np.int32)

        self.graph._retryer(
            self.lib.SampleNodes,
            self.ns_,
            c_int64(seed if seed is not None else random.getrandbits(64)),
            c_size_t(size),
            result_nodes.ctypes.data_as(POINTER(c_int64)),
            result_types.ctypes.data_as(POINTER(c_int32)),
        )

        return (result_nodes, result_types)

    def reset(self):
        """Reset sampler and free memory/disconnect from servers."""
        self.lib.ResetSampler(self.ns_)


class EdgeSampler:
    """Sampler to fetch edges from a graph."""

    def __init__(
        self, g: MemoryGraph, types: Union[List[int], int], kind: str = "weighted"
    ):
        """Create sampler from the graph.

        Args:
            g (MemoryGraph): graph to use for sampling.
            types (Union[List, int]): edge types to sample.
            kind (str, optional): sampling strategy. Defaults to "weighted".
        """
        if isinstance(types, int) and types == -1:
            types = []
            for tp in range(g.meta.edge_type_count):
                types.append(tp)

        types = _make_sorted_list(types)
        assert isinstance(types, List)

        types = list(
            filter(
                lambda tp: tp >= 0
                and tp < g.meta.edge_type_count
                and g.meta.edge_count_per_type[tp] > 0,
                types,
            )
        )

        assert (
            len(types) > 0
        ), "Sampler doesn't have any edges to select from the graph. Check graph.edge_count_per_type to inspect edge counts"
        assert isinstance(types, List)
        self.types = types
        self.es_ = _DEEP_GRAPH()
        self.graph = g
        self.lib = g.lib
        sampler_func = {
            "weighted": self.lib.CreateWeightedEdgeSampler,
            "uniform": self.lib.CreateUniformEdgeSampler,
            "withoutreplacement": self.lib.CreateUniformEdgeSamplerWithoutReplacement,
        }
        sampler_func[kind].argtypes = [
            POINTER(_DEEP_GRAPH),
            POINTER(_DEEP_GRAPH),
            c_size_t,
            POINTER(c_int32),
        ]
        sampler_func[kind].restype = c_int32
        sampler_func[kind].errcheck = _ErrCallback(  # type: ignore
            "create edge sampler"
        )
        sampler_func[kind].restype = c_int32
        TypeArray = c_int32 * len(types)
        types_array = TypeArray(*types)
        sampler_func[kind](
            byref(self.graph.g_), byref(self.es_), len(types), types_array
        )
        self._describe_clib_functions()

    def _describe_clib_functions(self):
        self.lib.SampleEdges.argtypes = [
            POINTER(_DEEP_GRAPH),
            c_int64,
            c_size_t,
            POINTER(c_int64),
            POINTER(c_int64),
            POINTER(c_int32),
        ]
        self.lib.SampleEdges.restype = c_int32
        self.lib.SampleEdges.errcheck = _ErrCallback("sample edges")  # type: ignore

        self.lib.ResetSampler.argtypes = [POINTER(_DEEP_GRAPH)]
        self.lib.ResetSampler.restype = c_int32
        self.lib.ResetSampler.errcheck = _ErrCallback(
            "reset edge sampler"
        )  # type: ignore

    def __del__(self):
        """Delete edge sampler."""
        self.reset()

    def sample(
        self, size: int, seed: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Perform sampling.

        Args:
            size (int): number of edges to sample.
            seed (int, optional): seed value to use for sampling. Defaults to random.getrandbits(64).

        Returns:
            Tuple[np.ndarray, np.ndarray]: a tuple of source, destination nodes for edges and their corresponding types.
        """
        result_src = np.empty(size, dtype=np.int64)
        result_dst = np.empty(size, dtype=np.int64)
        result_types = np.empty(size, dtype=np.int32)

        self.graph._retryer(
            self.lib.SampleEdges,
            self.es_,
            c_int64(seed if seed is not None else random.getrandbits(64)),
            c_size_t(size),
            result_src.ctypes.data_as(POINTER(c_int64)),
            result_dst.ctypes.data_as(POINTER(c_int64)),
            result_types.ctypes.data_as(POINTER(c_int32)),
        )

        return (result_src, result_dst, result_types)

    def reset(self):
        """Reset sampler and free memory/disconnect from servers."""
        self.lib.ResetSampler(self.es_)
