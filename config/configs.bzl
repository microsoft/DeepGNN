# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Configurations to build manylinux pip package"""

def _manylinux_config(name, compiler, sysroot = None):
    if cuda_version != None and rocm_version != None:
        fail("Specifying both cuda_version and rocm_version is not supported.")

    env = {
        "ABI_VERSION": "gcc",
        "ABI_LIBC_VERSION": "glibc_2.17",
        "BAZEL_COMPILER": compiler,
        "BAZEL_HOST_SYSTEM": "i686-unknown-linux-gnu",
        "BAZEL_TARGET_LIBC": "glibc_2.17",
        "BAZEL_TARGET_CPU": "k8",
        "BAZEL_TARGET_SYSTEM": "x86_64-unknown-linux-gnu",
        "CC_TOOLCHAIN_NAME": "linux_gnu_x86",
        "CC": compiler,
        "CLEAR_CACHE": "1",
        "HOST_CXX_COMPILER": compiler,
        "HOST_C_COMPILER": compiler,
    }


manylinux_config = _manylinux_config


def initialize_manylinux_configs():
    manylinux_config("manylinux", "gcc-11")
