# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import sys
import random
import tempfile
import pytest
import numpy as np
import numpy.testing as npt
from deepgnn.graph_engine.data.citation import Cora
from deepgnn.graph_engine import graph_ops
from deepgnn.graph_engine.snark.local import Client


def test_edge_sub_graph():
    random.seed(0)

    data_dir = tempfile.TemporaryDirectory()
    cl = Cora(data_dir.name)

    edges = np.array(
        [
            [0, 1, 0],
            [1, 2, 0],
        ]
    )

    subgraph = graph_ops.edge_sub_graph(cl, edges, [5, 5])
    npt.assert_equal(
        subgraph,
        np.array(
            [
                [0, 1, 0],
                [1, 2, 0],
                [1, 652, 0],
                [1, 654, 0],
                [2, 1, 0],
                [2, 332, 0],
                [2, 1454, 0],
                [2, 1666, 0],
                [2, 1986, 0],
                [652, 1, 0],
                [652, 470, 0],
                [654, 1, 0],
                [1666, 2, 0],
                [1666, 606, 0],
                [1666, 2381, 0],
                [1986, 1558, 0],
                [1986, 1859, 0],
                [1986, 1994, 0],
                [1986, 1995, 0],
            ]
        ),
    )


if __name__ == "__main__":
    sys.exit(
        pytest.main(
            [__file__, "--junitxml", os.environ["XML_OUTPUT_FILE"], *sys.argv[1:]]
        )
    )
