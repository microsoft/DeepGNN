# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

load("@rules_python//python:defs.bzl", "py_test")
load("//config:variables.bzl", "PLATFORM_DEFINES")
load("@rules_python//python:defs.bzl", "py_library")
load("@pip_deps//:requirements.bzl", "requirement")

py_library(
    name = "graph_engine_snark",
    srcs = [
        "_downloader.py",
        "_lib.py",
        "alias.py",
        "client.py",
        "convert.py",
        "meta_merger.py",
        "converter/__init__.py",
        "converter/process.py",
        "converter/writers.py",
        "converter/options.py",
        "decoders.py",
        "dispatcher.py",
        "distributed.py",
        "local.py",
        "meta.py",
        "server.py",
    ],
    data = ["//src/cc/lib:wrapper"],
    deps = [
        "//src/python/deepgnn/graph_engine:graph_engine",
    ],
    visibility = ["//visibility:public"],
)
