# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

load("@rules_python//python:defs.bzl", "py_test")
load("@pip_deps//:requirements.bzl", "requirement")

py_test(
    name = "notebook_test",
    srcs = ["notebook_test.py"],
    imports = ["../../src/python/"],
    main = "notebook_test.py",
    python_version = "PY3",
    srcs_version = "PY3",
    data = [
        "//docs/tutorials/tf:1_node_class.ipynb",
        "//docs/tutorials/tf:2_link_pred.ipynb",
        "//docs/tutorials/pytorch:1_node_class.ipynb",
        "//docs/tutorials/pytorch:2_data.ipynb",
        "//docs/tutorials/pytorch:3_link_pred.ipynb",
        "//docs/tutorials/other:debug_graph_data.ipynb",
    ],
    deps = [
        "//src/python/deepgnn/graph_engine:graph_engine",
        "//src/python/deepgnn/graph_engine/backends:graph_engine_backends",
        "//src/python/deepgnn/graph_engine/snark:graph_engine_snark",
        "//src/python/deepgnn/graph_engine/data:graph_engine_data",
        "//src/python/deepgnn/pytorch:deepgnn_pytorch",
        "//src/python/deepgnn/pytorch/common:deepgnn_pytorch_common",
        "//src/python/deepgnn/pytorch/encoding:deepgnn_pytorch_encoding",
        "//src/python/deepgnn/pytorch/modeling:deepgnn_pytorch_modeling",
        "//src/python/deepgnn/pytorch/nn:deepgnn_pytorch_nn",
        "//src/python/deepgnn/pytorch/training:deepgnn_pytorch_training",
        "//src/python/deepgnn/tf:deepgnn_tf",
        "//src/python/deepgnn/tf/common:deepgnn_tf_common",
        "//src/python/deepgnn/tf/nn:deepgnn_tf_nn",
        requirement("numpy"),
        requirement("matplotlib"),
        requirement("pytest"),
        requirement("scikit-learn"),
        requirement("torch"),
        requirement("tensorflow"),
        requirement("transformers"),
        requirement("nbval"),
        requirement("fsspec"),
        requirement("networkx"),
        requirement("tensorboard"),
        requirement("opencensus"),
        requirement("opencensus-context"),
        requirement("opencensus-ext-azure"),
        requirement("azure-datalake-store"),
        requirement("horovod"),
    ],
)
