# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Define graph API for GNN models."""

import abc
import fsspec
import numpy as np
from typing import Tuple, Union, Optional, List
from enum import IntEnum
from fsspec.utils import infer_storage_options


QueryOutput = Union[dict, tuple, np.ndarray, list]


class SamplingStrategy(IntEnum):
    """Strategies to sample node/edge from the graph engine."""

    Weighted = 1
    Random = 2  # RandomWithReplacement
    RandomWithoutReplacement = 3
    TopK = 4
    PPRGo = 5
    LastN = 6


class Graph(abc.ABC):
    """
    Implementation of the iterable dataset for GNN models.

    Iterator defines graph traversal type(node/edge) and
    additional graph operators are needed to implement GNN models.
    """

    @abc.abstractmethod
    def sample_nodes(
        self, size: int, node_types: Union[int, np.ndarray], strategy: SamplingStrategy
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Return a list of nodes with a specified type.

        Args:
        - size        -- number of nodes to sample.
        - node_types  -- types of nodes to sample.
        - strategy    -- sampling strategy.

        Returns:
            if node_types is `int`, return node_ids (an np.int64 array with length size).
            if node_types is `np.ndarray`, return node_ids, node_types (an np.int64 array, an np.int32 array with length size).
        """
        raise NotImplementedError

    @abc.abstractmethod
    def sample_edges(
        self, size: int, edge_types: Union[int, np.ndarray], strategy: SamplingStrategy
    ) -> np.ndarray:
        """
        Return a list of edges with a specified type.

        Args:
        - size        -- number of edges to sample.
        - edge_types  -- type of edges to sample.
        - strategy    -- sampling strategy.

        Returns an np.int64 array with length size.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def sample_neighbors(
        self,
        nodes: np.ndarray,
        edge_types: Union[int, np.ndarray],
        count: int = 10,
        strategy: str = "byweight",
        default_node: int = -1,
        default_weight: float = 0.0,
        default_edge_type: int = -1,
        alpha: float = 0.5,
        eps: float = 0.0001,
        timestamps: Union[List[int], np.ndarray] = None,
        return_edge_created_ts: bool = False,
    ) -> Union[
        Tuple[np.ndarray, np.ndarray, np.ndarray],
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    ]:
        """
        Sample node neighbors.

        There might be repetition of returned nodes, e.g. if count is greater than number of node neighbors.
        nodes -- array of nodes to select neighbors
        edge_types -- type of edges to use for selection.
        count -- fixed number of neighbors to select.
        strategy -- sampling strategy for neighbors.
        default_node -- default node id if the neighbor cannot be retrieved.
        default_weight -- default weight of the node's neighbor.
        default_edge_type -- default node edge of the node's neighbor.
        alpha -- ppr sampling teleport probability.
        eps -- stopping threshold for ppr sampling.
        timestamps -- timestamps to specify graph snapshot to sample neighbors for every node in a temporal graph.
        return_edge_created_ts -- if specified, timestamps will be added to the end of returned tuple.

        Returns a tuple of arrays nodes(np.uint64), weights(np.float), types(np.int32) and timestamps(np.int64)(if return_edge_created_ts is set) with shape [len(nodes), count]
        """
        raise NotImplementedError

    @abc.abstractmethod
    def random_walk(
        self,
        node_ids: np.ndarray,
        metapath: np.ndarray,
        walk_len: int,
        p: float,
        q: float,
        default_node: int = -1,
        timestamps: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Sample nodes via random walk.

        node_ids -- starting nodes
        metapath -- types of edges to sample
        walk_len -- number of steps to make
        p -- return parameter
        q -- in-out parameter
        default_node -- default node id if the neighbor cannot be retrieved
        timestamps -- timestamps to specify graph snapshot to sample neighbors for every node in a temporal graph.
        Returns starting and nodes visited during the walk
        """
        raise NotImplementedError

    @abc.abstractmethod
    def neighbors(
        self,
        nodes: np.ndarray,
        edge_types: Union[int, np.ndarray],
        timestamps: Optional[np.ndarray] = None,
        return_edge_created_ts: bool = False,
    ) -> Union[
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    ]:
        """
        Full list of node neighbors.

        nodes -- array of nodes to select neighbors
        edge_types -- type of edges to use for selection.
        timestamps -- timestamps to specify graph snapshot to retrieve neighbors for every node in a temporal graph.

        Returns a tuple of numpy arrays:
        -- neighbor ids per node, a one dimensional list of node ids
           concatenated in the same order as input nodes.
        -- weights for every neighbor, a one dimensional array
           of neighbor weights concatenated in the same order as input nodes.
        -- types of every neighbor.
        -- number of neighbors for every input node.
        -- timestamps of when edge connecting nodes was created (if return_edge_created_ts is set).
        """
        raise NotImplementedError

    @abc.abstractmethod
    def node_features(
        self,
        nodes: np.ndarray,
        features: np.ndarray,
        feature_type: np.dtype,
        timestamps: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Fetch node features.

        nodes -- array of nodes to fetch features from.
        features -- two dimensional int array where each row is [feature_id, feature_dim].
        feature_type -- type of the features to extract.
        timestamps -- timestamps to specify graph snapshot to fetch node features for every node in a temporal graph.

        Returns a blob array with feature values per node. The shape of the array is
        [len(nodes), sum(map(lambda f: f[1], features)))].
        """
        raise NotImplementedError

    @abc.abstractmethod
    def edge_features(
        self,
        edges: np.ndarray,
        features: np.ndarray,
        feature_type: np.dtype,
        timestamps: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Fetch edge features.

        edges -- array of triples [src, dst, type].
        features -- array of pairs describing features: [feature_id, feature_dim].
        feature_type -- type of features to extract.
        timestamps -- timestamps to specify graph snapshot to fetch edge features for every node in a temporal graph.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def node_types(self, nodes: np.ndarray) -> np.ndarray:
        """Fetch node types.

        nodes -- input array of nodes.
        Returns an array of types per each node.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def node_count(self, types: Union[int, np.ndarray]) -> int:
        """Return the number of nodes."""
        raise NotImplementedError

    @abc.abstractmethod
    def edge_count(self, types: Union[int, np.ndarray]) -> int:
        """Return the number of edges."""
        raise NotImplementedError


def get_fs(path: str):
    """Get fsspec filesystem object by path."""
    options = infer_storage_options(path)
    # Remove 'path'/'host' from kwargs because it is not supported by all filesystems
    if "path" in options:
        del options["path"]
    if "host" in options:
        del options["host"]
    return fsspec.filesystem(**options), options
