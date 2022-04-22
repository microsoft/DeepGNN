# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import sys
import random
import pytest
import deepgnn.graph_engine.snark.alias as alias
import os


def test_sanity_alias():
    random.seed(5)
    t = alias.Vose([0, 1, 2, 3], [1.0, 2.4, 0.5, 0.1])
    # expected ratios: [0.25, 0.6, 0.125, 0.025]
    counts = [0, 0, 0, 0]
    num_trials = 10000
    for _ in range(num_trials):
        counts[t.sample()] += 1
    assert counts[0] == 2454
    assert counts[1] == 6006
    assert counts[2] == 1279
    assert counts[3] == 261


if __name__ == "__main__":
    sys.exit(
        pytest.main(
            [__file__, "--junitxml", os.environ["XML_OUTPUT_FILE"], *sys.argv[1:]]
        )
    )
