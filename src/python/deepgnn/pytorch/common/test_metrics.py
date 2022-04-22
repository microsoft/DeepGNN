# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import pytest

import torch
import deepgnn.pytorch.common.metrics as metrics


@pytest.fixture
def mrr_test_tensors():
    # Return list of scores, labels, ascending expected value, descending expected value
    return [
        (
            torch.tensor([[1, 2.3]]),
            torch.tensor([[0, 1]]),  # Positive example in the end.
            0.5,  # Ascending expected value
            1.0,  # Descending expected value
        ),
        (
            torch.tensor([[1.0, 0.08, 0.01]]),
            torch.tensor([[0, 1, 0]]),  # Positive example in the middle.
            0.5,  # Ascending expected value
            0.5,  # Descending expected value
        ),
        (
            torch.tensor([[0.2, 0.8, 0.1]]),
            torch.tensor([[0, 0, 1]]),  # Positive example in the end.
            1.0,  # Ascending expected value
            (1.0 / 3.0),  # Descending expected value
        ),
        (
            torch.tensor([[0.2, 0.8, 0.1, 0.3, 0.4]]),
            torch.tensor([[1, 0, 0, 0, 0]]),  # Positive example in the front.
            (1.0 / 2.0),  # Ascending expected value
            (1.0 / 4.0),  # Descending expected value
        ),
    ]


def test_mrr_implementation(mrr_test_tensors):
    mrr_asc = metrics.MRR(rank_in_ascending_order=True)
    mrr_desc = metrics.MRR(rank_in_ascending_order=False)
    for scores, labels, asc_expected, desc_expected in mrr_test_tensors:
        asc_pred = mrr_asc.compute(scores, labels)
        assert asc_pred == asc_expected
        desc_pred = mrr_desc.compute(scores, labels)
        assert desc_pred == desc_expected
