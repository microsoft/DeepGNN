# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import abc
from deepgnn.graph_engine._base import Graph


class GraphEngineBackend(abc.ABC):
    """Interface class of all backends for graph engine."""

    @property
    def graph(self) -> Graph:
        """Get the graph client."""
        raise NotImplementedError

    def close(self):
        """Close backend object."""
        raise NotImplementedError
