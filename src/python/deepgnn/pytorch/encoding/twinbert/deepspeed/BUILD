# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
load("@rules_python//python:defs.bzl", "py_test")
load("//config:variables.bzl", "PLATFORM_DEFINES")
load("@rules_python//python:defs.bzl", "py_library")

py_library(
    name = "deepgnn_pytorch_encoding_twinbert_ds",
    srcs = [
        "__init__.py",
        "convert_bert_ckpt_to_deepspeed.py",
        "file_utils.py",
        "loss.py",
        "nvidia_modeling_no_apex.py",
        "nvidia_modeling.py",
    ],
    visibility = ["//visibility:public"],
)
