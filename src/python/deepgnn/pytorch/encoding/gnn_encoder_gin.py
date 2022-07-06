"""Various encoders implementations."""
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from deepgnn.graph_engine import Graph, FeatureType
from deepgnn import get_logger

class GINEncoder(nn.Module):
    """Encode a node's using 'convolutional' GIN approach."""

    def __init__(
        self,
        features,
        query_func,
        feature_dim: int,
        aggregator: nn.Module,
        num_sample: int,
        intermediate_dim: int,
        embed_dim: int = 256,
        edge_type: int = 0,
        activation_fn: Callable = F.relu,
        base_model=None,
    ):
        """Initialize GINEncoder.

        Args:
            features: callback used to generate node feature embedding.
            query_func: query funtion used to fetch data from graph engine.
            feature_dim: used to specify dimension of features when fetch data from graph engine.
            intermediate_dim: used to define the trainable weight metric. if there is a feature
                encoder, intermeidate_dim means the dimension of output of specific feature encoder,
                or it will be the same as feature_dim.
            embed_dim: output embedding dimension of GINEncoder.
        """
        super(GINEncoder, self).__init__()

        def query():
            pass

        def query_feature():
            pass

        def forward():
            pass
