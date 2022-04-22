# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Graph engine package contains key components such as
backend, dataset, samplers.

Put these components into a flatten namespace for easy usability.

e.g.

import deepgnn.graph_engine as ge

dataset = ge.DeepGNNDataset(
    sampler=ge.GENodeSampler,
    backend_options=ge.BackendOptions(args),
    ...
)

for i, data in enumerate(dataset):
    # train

"""

# flake8: noqa
from deepgnn.graph_engine._base import *
from deepgnn.graph_engine._adl_reader import (
    TextFileIterator,
    TextFileSplitIterator,
    AdlCredentialParser,
)
from deepgnn.graph_engine.samplers import *
from deepgnn.graph_engine import graph_ops
from deepgnn.graph_engine import multihop
from deepgnn.graph_engine import backends
from deepgnn.graph_engine.graph_dataset import *
from deepgnn.graph_engine.utils import define_param_graph_engine
