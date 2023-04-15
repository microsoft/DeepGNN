# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Snark clients implementation."""
import random
from typing import List, Tuple, Union, Dict

import numpy as np
import deepgnn.graph_engine.snark.client as client
from deepgnn.graph_engine._base import Graph, SamplingStrategy
from deepgnn import get_logger


_sampling_map: Dict[SamplingStrategy, str] = {
    SamplingStrategy.Weighted: "weighted",
    SamplingStrategy.Random: "uniform",
    SamplingStrategy.RandomWithoutReplacement: "withoutreplacement",
    SamplingStrategy.TopK: "topk",
    SamplingStrategy.PPRGo: "ppr-go",
    SamplingStrategy.LastN: "lastn",
}


class Client(Graph):
    """Wrapper for a snark graph client."""

    def __init__(
        self,
        path: str,
        partitions: List[int],
        storage_type: client.PartitionStorageType = client.PartitionStorageType.memory,
        config_path: str = "",
        stream: bool = False,
    ):
        """Provide a convenient wrapper around ctypes API of native graph."""
        self.logger = get_logger()
        if len(partitions) == 0:
            self.logger.info("Partitions not set, defaulting to [0]")
            partitions = [0]
        partitions_with_paths = [
            (path, partition) for partition in range(len(partitions))
        ]
        self.logger.info(
            f"Graph data path: {path}. Partitions {partitions}. Storage type {storage_type}. Config path {config_path}. Stream {stream}."
        )
        self.graph = client.MemoryGraph(
            path, partitions_with_paths, storage_type, config_path, stream
        )
        self.node_samplers: Dict[str, client.NodeSampler] = {}
        self.edge_samplers: Dict[str, client.EdgeSampler] = {}
        self.logger.info(
            f"Loaded snark graph. Node counts: {self.graph.meta.node_count_per_type}. Edge counts: {self.graph.meta.edge_count_per_type}"
        )

    def __check_types(self, types: Union[int, np.ndarray]) -> List[int]:
        if type(types) == int:
            return [types]  # type: ignore
        else:
            assert isinstance(types, np.ndarray)
            return types.flatten().tolist()

    def sample_nodes(
        self, size: int, node_types: Union[int, np.ndarray], strategy: SamplingStrategy
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Return an array of nodes with a specified type."""
        assert size > 0

        key = _sampling_map[strategy] + str(node_types)
        if key not in self.node_samplers:
            self.node_samplers[key] = client.NodeSampler(
                self.graph, self.__check_types(node_types), _sampling_map[strategy]
            )

        # Explicitly set random bits so users can have reproducable samples
        # by setting random.seed(#some_number) before calling this method.
        nodes, types = self.node_samplers[key].sample(size, random.getrandbits(64))
        if type(node_types) == int:
            return nodes
        return (nodes, types)

    def sample_edges(
        self, size: int, edge_types: Union[int, np.ndarray], strategy: SamplingStrategy
    ) -> np.ndarray:
        """Return an array of edges with a specified type."""
        assert size > 0

        key = str(edge_types)
        if key not in self.edge_samplers:
            self.edge_samplers[key] = client.EdgeSampler(
                self.graph, self.__check_types(edge_types), _sampling_map[strategy]
            )

        # Explicitly set random bits so users can have reproducable samples
        # by setting random.seed(#some_number) before calling this method.
        src, dst, tp = self.edge_samplers[key].sample(size, random.getrandbits(64))
        return np.stack([src, dst, tp], axis=1)

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
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Sample node neighbors."""
        if strategy == "byweight":
            result = self.graph.weighted_sample_neighbors(  # type: ignore
                nodes,
                self.__check_types(edge_types),
                count,
                default_node,
                default_weight,
                seed=int(random.getrandbits(64)),
                timestamps=timestamps,
            )
            return result[0], result[1], result[2], np.empty((1), dtype=np.int32)
        without_replacement = strategy == "randomwithoutreplacement"
        if strategy in ["random", "randomwithoutreplacement"]:
            result = self.graph.uniform_sample_neighbors(  # type: ignore
                without_replacement,
                nodes,
                self.__check_types(edge_types),
                count,
                default_node,
                default_edge_type,
                seed=int(random.getrandbits(64)),
                timestamps=timestamps,
            )
            return (
                result[0],
                np.empty(result[0].shape, dtype=np.float32),
                result[1],
                np.empty((1), dtype=np.int32),
            )
        if strategy == "ppr-go":
            result = self.graph.ppr_neighbors(  # type: ignore
                nodes,
                self.__check_types(edge_types),
                count,
                alpha,
                eps,
                default_node,
                default_weight,
                timestamps=timestamps,
            )
            return (
                result[0],
                result[1],
                np.empty(0, dtype=np.int32),
                np.empty(0, dtype=np.int32),
            )

        if strategy == "lastn":
            return self.graph.lastn_neighbors(  # type: ignore
                nodes,
                self.__check_types(edge_types),
                timestamps=timestamps,
                count=count,
                default_node=default_node,
                default_weight=default_weight,
                default_edge_type=default_edge_type,
            )

        raise NotImplementedError(f"Unknown strategy type {strategy}")

    def node_features(
        self,
        nodes: np.ndarray,
        features: np.ndarray,
        feature_type: np.dtype,
        timestamps: Union[List[int], np.ndarray] = None,
    ) -> np.ndarray:
        """Fetch node features."""
        assert len(features.shape) == 2
        assert features.shape[-1] == 2
        for feature in features:
            if feature[0] >= self.graph.meta._node_feature_count:
                self.logger.error(
                    f"Requesting feature with id #{feature[0]} that is larger than number of the node features {self.graph.meta._node_feature_count} in the graph"
                )

        return self.graph.node_features(
            nodes, features, feature_type, timestamps=timestamps
        )

    def random_walk(
        self,
        node_ids: np.ndarray,
        metapath: np.ndarray,
        walk_len: int,
        p: float,
        q: float,
        default_node: int = -1,
        timestamps: Union[List[int], np.ndarray] = None,
    ) -> np.ndarray:
        """
        Sample nodes via random walk.

        :node_ids -- starting nodes
        :metapath -- types of edges to sample on every step
        :walk_len -- number of steps to make
        :p -- return parameter
        :q -- in-out parameter
        Returns starting and nodes visited during the walk
        """
        return self.graph.random_walk(
            node_ids,
            self.__check_types(metapath),
            walk_len,
            p,
            q,
            default_node,
            seed=random.getrandbits(64),
            timestamps=timestamps,
        )

    def neighbors(
        self,
        nodes: np.ndarray,
        edge_types: Union[int, np.ndarray],
        timestamps: Union[List[int], np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Fetch full information about node neighbors."""
        return self.graph.neighbors(
            nodes, self.__check_types(edge_types), timestamps=timestamps
        )

    def neighbor_count(
        self,
        nodes: np.ndarray,
        edge_types: Union[int, np.ndarray],
        timestamps: Union[List[int], np.ndarray] = None,
    ) -> np.ndarray:
        """Fetch node degrees."""
        return self.graph.neighbor_counts(
            nodes, self.__check_types(edge_types), timestamps=timestamps
        )

    def node_types(self, nodes: np.ndarray) -> np.ndarray:
        """Fetch node types."""
        return self.graph.node_types(nodes, -1)

    def edge_features(
        self,
        edges: np.ndarray,
        features: np.ndarray,
        feature_type: np.dtype,
        timestamps: Union[List[int], np.ndarray] = None,
    ) -> np.ndarray:
        """Fetch edge features."""
        edges = np.array(edges, dtype=np.int64)
        features = np.array(features, dtype=np.int32)
        assert len(features.shape) == 2
        assert features.shape[-1] == 2
        for feature in features:
            if feature[0] >= self.graph.meta._edge_feature_count:
                self.logger.error(
                    f"Requesting feature with id #{feature[0]} that is larger than number of the edge features {self.graph.meta._edge_feature_count} in the graph"
                )

        return self.graph.edge_features(
            np.copy(edges[:, 0]),
            np.copy(edges[:, 1]),
            np.copy(edges[:, 2]),
            features,
            feature_type,
            timestamps=timestamps,
        )

    def node_count(self, types: Union[int, np.ndarray]) -> int:
        """Return node count."""
        if isinstance(types, int):
            return self.graph.get_node_type_count([types])

        return self.graph.get_node_type_count(types.tolist())

    def edge_count(self, types: Union[int, np.ndarray]) -> int:
        """Return edge count."""
        if isinstance(types, int):
            return self.graph.get_edge_type_count([types])

        return self.graph.get_edge_type_count(types.tolist())

    def reset(self):
        """Delete client."""
        self.graph.reset()
        for _, sampler in self.node_samplers.items():
            sampler.reset()
        for _, sampler in self.edge_samplers.items():
            sampler.reset()
