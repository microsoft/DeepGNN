# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

load("@rules_python//python:defs.bzl", "py_binary", "py_test")
load("@doc_deps//:requirements.bzl", doc_requirement = "requirement")
load("@pip_deps//:requirements.bzl", "requirement")

common_deps = [
    "//src/python/deepgnn:deepgnn_ge_wheel_library",
    requirement("azure-datalake-store"),
    requirement("fsspec"),
    requirement("networkx"),
    requirement("numpy"),
    requirement("opencensus"),
    requirement("opencensus-context"),
    requirement("opencensus-ext-azure"),
    requirement("tenacity"),
    requirement("scipy"),
]

py_test(
    name = "sql_shard",
    srcs = ["doctest_template.py"],
    args = ["advanced/sql_shard.rst"],
    data = [
        "advanced/sql_shard.rst",
        "//src/cc/lib/distributed:service_proto",
    ],
    imports = ["../src/python/"],
    main = "doctest_template.py",
    python_version = "PY3",
    srcs_version = "PY3",
    deps = common_deps + [
        requirement("grpcio"),
        requirement("grpcio-tools"),
    ],
)

py_test(
    name = "custom_decoder",
    srcs = ["doctest_template.py"],
    args = ["graph_engine/custom_decoder.rst"],
    data = [
        "graph_engine/custom_decoder.rst",
    ],
    imports = ["../src/python/"],
    main = "doctest_template.py",
    python_version = "PY3",
    srcs_version = "PY3",
    deps = common_deps + [requirement("avro-python3")],
)

py_test(
    name = "from_networkx",
    srcs = ["doctest_template.py"],
    args = ["graph_engine/from_networkx.rst"],
    data = [
        "graph_engine/from_networkx.rst",
    ],
    imports = ["../src/python/"],
    main = "doctest_template.py",
    python_version = "PY3",
    srcs_version = "PY3",
    deps = common_deps,
)

py_test(
    name = "spark_converter",
    srcs = ["doctest_template.py"],
    args = ["graph_engine/spark_converter.rst"],
    data = [
        "graph_engine/spark_converter.rst",
    ],
    imports = ["../src/python/"],
    main = "doctest_template.py",
    python_version = "PY3",
    srcs_version = "PY3",
    deps = common_deps + [
        requirement("pyspark"),
    ],
)

py_test(
    name = "temporal_graph",
    srcs = ["doctest_template.py"],
    args = ["graph_engine/temporal.rst"],
    data = [
        "graph_engine/temporal.rst",
    ],
    imports = ["../src/python/"],
    main = "doctest_template.py",
    python_version = "PY3",
    srcs_version = "PY3",
    deps = common_deps + [
        requirement("pyspark"),
    ],
)

common_torch_deps = common_deps + [
    requirement("scikit-learn"),
    requirement("torch"),
    requirement("torch-sparse"),
    requirement("torch-scatter"),
    requirement("torch-cluster"),
    requirement("torch_geometric"),
]

py_test(
    name = "torch_node_class",
    srcs = ["doctest_template.py"],
    args = ["torch/node_class.rst"],
    data = [
        "torch/node_class.rst",
    ],
    imports = ["../src/python/"],
    main = "doctest_template.py",
    python_version = "PY3",
    srcs_version = "PY3",
    deps = common_torch_deps,
)

py_test(
    name = "torch_link_prediction",
    srcs = ["doctest_template.py"],
    args = ["torch/link_pred.rst"],
    data = [
        "torch/link_pred.rst",
    ],
    imports = ["../src/python/"],
    main = "doctest_template.py",
    python_version = "PY3",
    srcs_version = "PY3",
    tags = ["manual"], # CI runners run out of storage for this test.
    deps = common_torch_deps + [
        requirement("littleutils"),
        requirement("ogb"),
        requirement("outdated"),
        requirement("mpmath"),
        requirement("pytz"),
        requirement("pandas"),
        requirement("sympy"),
    ],
)

py_test(
    name = "torch_distrib",
    srcs = ["doctest_template.py"],
    args = ["torch/distrib.rst"],
    data = [
        "torch/distrib.rst",
    ],
    imports = ["../src/python/"],
    main = "doctest_template.py",
    python_version = "PY3",
    srcs_version = "PY3",
    deps = common_torch_deps + [
        requirement("ray"),
    ],
)

common_tf_deps = [
    "//src/python/deepgnn/graph_engine/backends:graph_engine_backends",
    "//src/python/deepgnn/graph_engine/data:graph_engine_data",
    "//src/python/deepgnn/graph_engine/snark:graph_engine_snark",
    "//src/python/deepgnn/tf:deepgnn_tf",
    "//src/python/deepgnn/tf/common:deepgnn_tf_common",
    "//src/python/deepgnn/tf/nn:deepgnn_tf_nn",
    requirement("azure-datalake-store"),
    requirement("fsspec"),
    requirement("numpy"),
    requirement("networkx"),
    requirement("matplotlib"),
    requirement("opencensus"),
    requirement("opencensus-context"),
    requirement("opencensus-ext-azure"),
    requirement("scikit-learn"),
    requirement("tensorflow"),
    requirement("tenacity"),
]

py_test(
    name = "tf_node_class",
    srcs = ["doctest_template.py"],
    args = ["tf/node_class.rst"],
    data = [
        "tf/node_class.rst",
    ],
    imports = ["../src/python/"],
    main = "doctest_template.py",
    python_version = "PY3",
    srcs_version = "PY3",
    target_compatible_with = [
        "@platforms//os:linux",
    ],
    deps = common_tf_deps,
)

py_test(
    name = "tf_link_prediction",
    srcs = ["doctest_template.py"],
    args = ["tf/link_pred.rst"],
    data = [
        "tf/link_pred.rst",
    ],
    imports = ["../src/python/"],
    main = "doctest_template.py",
    python_version = "PY3",
    srcs_version = "PY3",
    target_compatible_with = [
        "@platforms//os:linux",
    ],
    deps = common_tf_deps,
)

py_test(
    name = "tf_ray_usage",
    srcs = ["doctest_template.py"],
    args = ["tf/ray_usage.rst"],
    data = [
        "tf/ray_usage.rst",
    ],
    env = {"PYTHONHASHSEED": "0"},
    imports = ["../src/python/"],
    main = "doctest_template.py",
    python_version = "PY3",
    srcs_version = "PY3",
    tags = ["exclusive"],
    target_compatible_with = [
        "@platforms//os:linux",
    ],
    deps = common_tf_deps + [
        requirement("ray"),
        requirement("pyarrow"),
        requirement("torch"),
        requirement("setuptools"),
    ],
)

py_binary(
    name = "make_docs",
    srcs = [
        "make_docs.py",
    ],
    data = [
        "advanced/hdfs.rst",
        "advanced/index.rst",
        "advanced/sql_shard.rst",
        "conf.py",
        "graph_engine/data_spec.rst",
        "graph_engine/from_networkx.rst",
        "graph_engine/index.rst",
        "graph_engine/overview.rst",
        "graph_engine/spark_converter.rst",
        "index.rst",
        "tf/index.rst",
        "tf/link_pred.rst",
        "tf/node_class.rst",
        "torch/index.rst",
        "torch/link_pred.rst",
        "torch/node_class.rst",
    ],
    target_compatible_with = [
        "@platforms//os:linux",
    ],
    deps = [
        doc_requirement("alabaster"),
        doc_requirement("sphinx"),
        doc_requirement("docutils"),
        doc_requirement("imagesize"),
        doc_requirement("importlib-metadata"),
        doc_requirement("Jinja2"),
        doc_requirement("packaging"),
        doc_requirement("Pygments"),
        doc_requirement("pyparsing"),
        doc_requirement("requests"),
        doc_requirement("six"),
        doc_requirement("snowballstemmer"),
        doc_requirement("sphinxcontrib-websupport"),
        doc_requirement("sphinxcontrib-serializinghtml"),
        doc_requirement("sphinxcontrib-applehelp"),
        doc_requirement("sphinxcontrib-devhelp"),
        doc_requirement("sphinxcontrib-jsmath"),
        doc_requirement("sphinxcontrib-htmlhelp"),
        doc_requirement("sphinxcontrib-qthelp"),
        doc_requirement("sphinx-copybutton"),
    ],
)
