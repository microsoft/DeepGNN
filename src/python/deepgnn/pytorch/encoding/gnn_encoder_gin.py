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
        base_model=None,
    ):
        """Initialize GINEncoder.
        """
        super(GINEncoder, self).__init__()

        def query():
            pass

        def query_feature():
            pass

        def forward():
            pass
