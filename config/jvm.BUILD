# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

load("@rules_foreign_cc//foreign_cc:defs.bzl", "cmake")

cc_import(
  name = "jvm",
  shared_library = "jre/lib/amd64/server/libjvm.so",
  visibility = ["//visibility:public"],
)

filegroup(
  name = "jvm_py",
  srcs = [
      "jre/lib/amd64/server/libjvm.so",
  ],
  visibility = ["//visibility:public"],
)
