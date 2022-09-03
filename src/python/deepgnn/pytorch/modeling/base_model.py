# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Base classes for torch models."""
from typing import Optional, Tuple, Union, IO

import torch
import torch.nn as nn
import numpy as np

from torch.autograd import Variable
from deepgnn.graph_engine import Graph, FeatureType, SamplingStrategy, QueryOutput
from deepgnn.pytorch.encoding.feature_encoder import FeatureEncoder
from deepgnn.pytorch.common.metrics import BaseMetric
from deepgnn.graph_engine.samplers import BaseSampler
from deepgnn import get_logger


class BaseModel(nn.Module):
    """Define a base class for torch GNNs."""

    def __init__(
        self,
        feature_type: FeatureType,
        feature_idx: int,
        feature_dim: int,
        feature_enc: Optional[FeatureEncoder],
    ):
        """Initialize common fields.

        Args:
            feature_type: feature type used for graph querying.
            feature_idx: feature index used for graph querying.
            feature_dim: feature dimension used for graph querying.
            feature_enc: feature encoder used to encode input raw feature(mainly for binary) to embedding.
        """
        super(BaseModel, self).__init__()

        get_logger().info(
            f"[BaseModel] feature_type: {feature_type}, feature_idx:{feature_idx}, feature_dim:{feature_dim}."
        )
        self.feature_type = feature_type
        self.feature_idx = feature_idx

        # If feature_enc is valid, overwrite self.feature_dim with feature_enc.feature_dim.
        self.feature_dim = feature_enc.feature_dim if feature_enc else feature_dim
        self.feature_enc = feature_enc
        self.sampler: Optional[BaseSampler] = None
        self.xent = nn.CrossEntropyLoss()
        self.metric: BaseMetric

    def get_score(self, context: QueryOutput) -> torch.Tensor:
        """Evaluate model."""
        raise NotImplementedError

    def get_embedding(self, context: QueryOutput) -> torch.Tensor:
        """
        Get embedding.

        Data is array of [node_id, node_type]:
        [
        [node_id, node_type],
        [node_id, node_type],
        [node_id, node_type],
        ]
        """
        return self.get_score(context)

    def output_embedding(
        self, output: IO[str], context: QueryOutput, embeddings: torch.Tensor
    ):
        """Dump embeddings to a file."""
        embeddings = embeddings.data.cpu().numpy()
        if isinstance(context, dict):
            inputs = context["inputs"].squeeze(0)  # type: ignore
        elif isinstance(context, torch.Tensor):
            inputs = context.squeeze(0)  # type: ignore
        else:
            raise TypeError("Invalid input type.")
        embedding_strs = []
        for k in range(len(embeddings)):
            embedding_strs.append(
                str(inputs[k].cpu().numpy())
                + " "
                + " ".join([str(embeddings[k][x]) for x in range(len(embeddings[k]))])
                + "\n"
            )
        output.writelines(embedding_strs)

    def metric_name(self) -> str:
        """Metric used for model evaluation."""
        return self.metric.name() if self.metric is not None else ""

    def compute_metric(self, preds, labels) -> torch.Tensor:
        """Stub for metric evaluation."""
        if self.metric is not None:
            preds = torch.unsqueeze(torch.cat(preds, 0), 1)
            labels = torch.unsqueeze(torch.cat(labels, 0), 1).type(preds.dtype)
            return self.metric.compute(preds, labels)
        return torch.tensor(0.0)

    def query(self, graph: Graph, inputs: np.ndarray) -> QueryOutput:
        """Query graph engine to fetch graph data for model execution.

        This function will be invoked by prefetch. Args:
            graph: Graph to query.
            inputs: nested numpy array dictionary.
        """
        raise NotImplementedError

    def transform(self, context: dict):
        """Perform necessary transformation after fetching data from graph engine.

        This function will be invoked by prefetch. Args:
            context: nested numpy array dictionary.
        """
        if self.feature_enc:
            self.feature_enc.transform(context)

    def encode_feature(self, context: dict):
        """Encode feature vectors."""
        if self.feature_enc:
            self.feature_enc.forward(context)

    def forward(
        self, context: QueryOutput
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Execute common forward operation for all models.

        Args:
            context: nested tensor dictionary.
        """
        raise NotImplementedError


class BaseSupervisedModel(BaseModel):
    """Define a base class for supervised models."""

    def __init__(
        self,
        feature_type: FeatureType,
        feature_idx: int,
        feature_dim: int,
        feature_enc: Optional[FeatureEncoder],
    ):
        """Initialize common fields."""
        super(BaseSupervisedModel, self).__init__(
            feature_type=feature_type,
            feature_idx=feature_idx,
            feature_dim=feature_dim,
            feature_enc=feature_enc,
        )

    def _loss_inner(
        self, context: QueryOutput
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Cross entropy loss for a list of nodes."""
        if isinstance(context, dict):
            labels = context["label"].squeeze()  # type: ignore
        elif isinstance(context, torch.Tensor):
            labels = context.squeeze()  # type: ignore
        else:
            raise TypeError("Invalid input type.")
        device = labels.device

        # TODO(chaoyl): Due to the bug of pytorch argmax, we have to copy labels to numpy for argmax
        # then copy back to Tensor. The fix has been merged to pytorch master branch but not included
        # in latest stable version. Revisit this part after updating pytorch with the fix included.
        # issue: https://github.com/pytorch/pytorch/issues/32343
        # fix: https://github.com/pytorch/pytorch/pull/37864
        labels = labels.cpu().numpy().argmax(1)
        scores = self.get_score(context)
        return (
            self.xent(
                scores,
                Variable(torch.tensor(labels.squeeze(), dtype=torch.int64).to(device)),
            ),
            scores.argmax(dim=1),
            torch.tensor(labels.squeeze(), dtype=torch.int64),
        )

    def forward(
        self, context: QueryOutput
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return cross entropy loss."""
        return self._loss_inner(context)


class BaseUnsupervisedModel(BaseModel):
    """Define a base class for unsupervised models."""

    def __init__(
        self,
        feature_type: FeatureType,
        feature_idx: int,
        feature_dim: int,
        feature_enc: Optional[FeatureEncoder],
    ):
        """Initialize common fields."""
        super(BaseUnsupervisedModel, self).__init__(
            feature_type=feature_type,
            feature_idx=feature_idx,
            feature_dim=feature_dim,
            feature_enc=feature_enc,
        )

    def get_neg_node(
        self, graph: Graph, num_negs: int, neg_type: Union[int, np.ndarray]
    ) -> np.ndarray:
        """Fetch negative examples, random nodes in a graph."""
        output = graph.sample_nodes(num_negs, neg_type, SamplingStrategy.Weighted)
        if isinstance(output, tuple):
            output = output[0]
        return output

    def get_pos_node(
        self,
        graph: Graph,
        nodes: np.ndarray,
        edge_types: Union[int, np.ndarray],
        count: int = 1,
    ) -> np.ndarray:
        """Return positive examples, node neighbors."""
        return graph.sample_neighbors(nodes, edge_types, count)[0]
