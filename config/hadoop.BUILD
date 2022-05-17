# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

cc_import(
  name = "hadoop_so",
  shared_library = "lib/native/libhdfs.so",
  visibility = ["//visibility:public"],
)

cc_library(
  name = "hadoop",
  deps = [
    ":hadoop_so",
    "@jvm//:jvm",
  ],
  data = [":."],
  visibility = ["//visibility:public"],
)

cc_import(  # For #include "hdfs.h"
  name = "hadoop_include",
  hdrs = ["include/hdfs.h"],
  shared_library = "lib/native/libhdfs.so.0.0.0",
  visibility = ["//visibility:public"],
)

genrule(
  name = "gen_hadoop_py",
  srcs = [],
  outs = [
    "hadoop_py.py"
  ],
  cmd = "echo '' > $(@D)/hadoop_py.py",
  visibility = ["//visibility:public"],
)

py_binary(
  name = "hadoop_py",
  srcs = ["hadoop_py.py"],
  data = ["lib/native/libhdfs.so", "@jvm//:jvm_py"],
  deps = [],
  visibility = ["//visibility:public"],
)
