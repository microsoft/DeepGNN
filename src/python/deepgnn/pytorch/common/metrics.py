# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Metrics implementation for graphsage models."""

import torch
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score


class BaseMetric(object):
    """Base class for metrics."""

    def name(self) -> str:
        """Return name of the metric."""
        return self.__class__.__name__

    def compute(self, scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute metric value based on model scores and expected labels."""
        raise NotImplementedError


class F1Score(BaseMetric):
    """F1 score implementation."""

    def compute(self, scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Passthrough to scikit."""
        return torch.tensor(
            f1_score(labels.squeeze(), scores.detach().cpu().numpy(), average="micro")
        )


class Accuracy(BaseMetric):
    """Accuracy classification score."""

    def compute(self, scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Passthrough to scikit."""
        return torch.tensor(
            accuracy_score(y_true=labels.cpu(), y_pred=scores.detach().cpu().numpy())
        )


class MRR(BaseMetric):
    """MRR score implementation."""

    def __init__(self, rank_in_ascending_order: bool = False):
        """
        Initialize MRR metric.

        rank_in_ascending_order:
          Should we get the rank in the ascending order or
          descending order, if set to True will calculate
          the rank in ascending order.
        """
        super().__init__()
        self.rank_in_ascending_order = rank_in_ascending_order

    def compute(self, scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute metric based on logit scores."""
        assert len(scores.shape) > 1
        assert scores.size() == labels.size()

        size = scores.shape[-1]
        if self.rank_in_ascending_order:
            scores = -1 * scores
        _, indices_of_ranks = torch.topk(scores, k=size)
        _, ranks = torch.topk(-indices_of_ranks, k=size)
        return torch.mean(
            torch.reciprocal(
                torch.matmul(ranks.float(), torch.transpose(labels, -2, -1).float()) + 1
            )
        )


class ROC(BaseMetric):
    """ROC score implementation with scikit."""

    def compute(self, scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute metric value based on model scores and expected labels."""
        return torch.tensor(roc_auc_score(labels.cpu(), scores.cpu().detach().numpy()))
