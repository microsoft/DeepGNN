# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Define graph API for GNN models."""

import fsspec
import numpy as np
from typing import Tuple, Union, Iterator
from enum import Enum, IntEnum
from fsspec.utils import infer_storage_options


QueryOutput = Union[dict, tuple, np.ndarray, list]


class FeatureType(Enum):
    """Feature types to fetch from a graph engine."""

    FLOAT = 1
    INT64 = 2
    BINARY = 3


class SamplingStrategy(IntEnum):
    """Strategies to sample node/edge from the graph engine."""

    Weighted = 1
    Random = 2  # RandomWithReplacement
    RandomWithoutReplacement = 3
    TopK = 4


class Graph:
    """
    Implementation of the iterable dataset for GNN models.

    Iterator defines graph traversal type(node/edge) and
    additional graph operators are needed to implement GNN models.
    """

    ALL_NODE_TYPE = -1
    ALL_EDGE_TYPE = -1

    def __iter__(self) -> Iterator:
        """Node or Edge iterator."""
        raise NotImplementedError

    def __len__(self) -> int:
        """Return the number of nodes or edges in the graph."""
        raise NotImplementedError

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

    def sample_neighbors(
        self,
        nodes: np.ndarray,
        edge_types: Union[int, np.ndarray],
        count: int = 10,
        strategy: str = "byweight",
        default_node: int = -1,
        default_weight: float = 0.0,
        default_node_type: int = -1,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Sample node neighbors.

        There might be repetition of returned nodes, e.g. if count is greater than number of node neighbors.
        nodes -- array of nodes to select neighbors
        edge_types -- type of edges to use for selection.
        count -- fixed number of neighbors to select.
        strategy -- sampling strategy for neighbors.
        default_node -- default node id if the neighbor cannot be retrieved.
        default_weight -- default weight of the node's neighbor.
        default_node_type -- default node type of the node's neighbor.

        Returns a tuple of arrays nodes(np.uint64), weights(np.float) and types(np.int) with shape [len(nodes), count]
        """
        raise NotImplementedError

    def random_walk(
        self,
        node_ids: np.ndarray,
        metapath: np.ndarray,
        walk_len: int,
        p: float,
        q: float,
        default_node: int = -1,
    ) -> np.ndarray:
        """
        Sample nodes via random walk.

        node_ids -- starting nodes
        metapath -- types of edges to sample
        walk_len -- number of steps to make
        p -- return parameter
        q -- in-out parameter
        default_node -- default node id if the neighbor cannot be retrieved
        Returns starting and nodes visited during the walk
        """
        raise NotImplementedError

    def neighbors(
        self, nodes: np.ndarray, edge_types: Union[int, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Full list of node neighbors.

        nodes -- array of nodes to select neighbors
        edge_types -- type of edges to use for selection.

        Returns a tuple of numpy arrays:
        -- neighbor counts per node, with the shape [len(nodes)]
        -- neighbor ids per node, a one dimensional list of node ids
           concatenated in the same order as input nodes.
        -- weights for every neighbor, a one dimensional array
           of neighbor weights concatenated in the same order as input nodes.
        -- types of every neighbor.
        """
        raise NotImplementedError

    def node_features(
        self, nodes: np.ndarray, features: np.ndarray, feature_type: FeatureType
    ) -> np.ndarray:
        """Fetch node features.

        nodes -- array of nodes to fetch features from.
        features -- two dimensional int array where each row is [feature_id, feature_dim].
        feature_type -- type of the features to extract.

        Returns a blob array with feature values per node. The shape of the array is
        [len(nodes), sum(map(lambda f: f[1], features)))].
        """
        raise NotImplementedError

    def edge_features(
        self, edges: np.ndarray, features: np.ndarray, feature_type: FeatureType
    ) -> np.ndarray:
        """Fetch edge features.

        edges -- array of triples [src, dst, type].
        features -- array of pairs describing features: [feature_id, feature_dim].
        feature_type -- type of features to extract.
        """
        raise NotImplementedError

    def node_types(self, nodes: np.ndarray) -> np.ndarray:
        """Fetch node types.

        nodes -- input array of nodes.
        Returns an array of types per each node.
        """
        raise NotImplementedError

    def node_count(self, types: Union[int, np.ndarray]) -> int:
        """Return the number of nodes."""
        raise NotImplementedError

    def edge_count(self, types: Union[int, np.ndarray]) -> int:
        """Return the number of edges."""
        raise NotImplementedError


def get_fs(path: str):
    """Get fsspec filesystem object by path."""
    options = infer_storage_options(path)
    # Remove 'path' from kwargs because it is not supported by all filesystems
    del options["path"]
    return fsspec.filesystem(**options), options
