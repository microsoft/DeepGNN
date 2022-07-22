# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Alias table generators."""
import random
import typing

import numpy as np


class Vose:
    """Generate alias tables with Vose method."""

    def __init__(self, elements: typing.List, weights: np.ndarray):
        """Create alias tables for weight sampling.

        Args:
            elements (typing.List): elements to sample
            weights (np.array): corresponging elements weights
        """
        self.elements = elements
        weights = np.multiply(weights, len(weights) / np.sum(weights))
        self.alias = np.empty(len(elements), dtype=np.uint64)
        self.prob = np.empty(len(elements), dtype=np.float32)
        self._generate_table(weights)

    def _generate_table(self, weights: np.ndarray):
        small = []
        large = []
        for i, w in enumerate(weights):
            if w < 1:
                small.append(i)
            else:
                large.append(i)

        while small and large:
            small_element = small.pop()
            large_element = large.pop()
            self.alias[small_element] = large_element
            self.prob[small_element] = weights[small_element]

            weights[large_element] = (
                weights[large_element] + weights[small_element]
            ) - 1
            if weights[large_element] < 1:
                small.append(large_element)
            else:
                large.append(large_element)

        while large:
            self.prob[large.pop()] = 1

        while small:
            self.prob[small.pop()] = 1

    def sample(self) -> typing.Any:
        """Sample from alias tables.

        Returns:
            typing.Any: element from the original list
        """
        n = random.randrange(len(self.alias))
        if random.uniform(0, 1) > self.prob[n]:
            n = self.alias[n]
        return self.elements[n]
