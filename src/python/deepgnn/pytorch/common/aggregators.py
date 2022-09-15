# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Set of modules for aggregating embeddings of neighbors."""

from typing import Callable

import torch
import torch.nn as nn


class MeanAggregator(nn.Module):
    """Aggregates a node's embeddings using mean of neighbors' embeddings."""

    def __init__(self, features: Callable[[torch.Tensor], torch.Tensor]):
        """
        Initialize the aggregator for a specific graph.

        features -- function mapping LongTensor of node ids to FloatTensor of feature values.
        """
        super(MeanAggregator, self).__init__()

        self.features = features

    def forward(self, neighs: torch.Tensor, node_count: int) -> torch.Tensor:
        """
        Propagate node features to NN Layer.

        neighs --- context of neighbors with a shape
        """
        neigh_feats = self.features(neighs)

        nb_count = int(neigh_feats.shape[0] / node_count)
        fv_by_node = neigh_feats.view(node_count, nb_count, neigh_feats.shape[-1])
        return fv_by_node.mean(1)
