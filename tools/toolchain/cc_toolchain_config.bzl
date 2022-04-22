# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Toolchain for gcc-11 compiler to link against glibc-2.17.
"""
load("@bazel_tools//tools/build_defs/cc:action_names.bzl", "ACTION_NAMES")
load(
    "@bazel_tools//tools/cpp:cc_toolchain_config_lib.bzl",
    "feature",
    "flag_group",
    "flag_set",
    "tool_path",
)

all_link_actions = [
    ACTION_NAMES.cpp_link_executable,
    ACTION_NAMES.cpp_link_dynamic_library,
    ACTION_NAMES.cpp_link_nodeps_dynamic_library,
]

all_compile_actions = [
    ACTION_NAMES.cpp_compile,
    ACTION_NAMES.cpp_module_compile,
]

def _impl(ctx):
    tool_paths = [
        tool_path(
            name = "gcc",
            path = "/dt11/usr/bin/gcc",
        ),
        tool_path(
            name = "ld",
            path = "/usr/bin/gold",
        ),
        tool_path(
            name = "ar",
            path = "/dt11//usr/bin/gcc-ar",
        ),
        tool_path(
            name = "cpp",
            path = "/dt11/usr/bin/g++",
        ),
        tool_path(
            name = "gcov",
            path = "/bin/false",
        ),
        tool_path(
            name = "nm",
            path = "/bin/false",
        ),
        tool_path(
            name = "objdump",
            path = "/bin/false",
        ),
        tool_path(
            name = "strip",
            path = "/bin/false",
        ),
    ]

    features = [
        feature(
            name = "default_compile_flags",
            enabled = True,
            flag_sets = [
                flag_set(
                    actions = all_compile_actions,
                    flag_groups = ([
                        flag_group(
                            flags = [
                                "-std=c++20",
                            ],
                        ),
                    ]),
                ),
            ],
        ),
        feature(
            name = "default_linker_flags",
            enabled = True,
            flag_sets = [
                flag_set(
                    actions = all_link_actions,
                    flag_groups = ([
                        flag_group(
                            flags = [
                                "-lstdc++",
                                "-Wl,--rpath=/dt11/usr/lib64",
                                "-Wl,--dynamic-linker='/dt11/usr/lib/ld-linux-x86-64.so.2'",
                            ],
                        ),
                    ]),
                ),
            ],
        ),
    ]

    return cc_common.create_cc_toolchain_config_info(
        ctx = ctx,
        features = features,
        cxx_builtin_include_directories = [
            "%sysroot%/usr/lib/gcc/x86_64-pc-linux-gnu/11/include",
            "%sysroot%/usr/lib/gcc/x86_64-pc-linux-gnu/11/include-fixed",
            "%sysroot%/usr/include",
            "/usr/include/",
        ],
        toolchain_identifier = "local",
        host_system_name = "local",
        target_system_name = "local",
        target_cpu = "k8",
        target_libc = "glibc-2.17.0",
        compiler = "gcc",
        abi_version = "unknown",
        abi_libc_version = "gcc-11.2",
        tool_paths = tool_paths,
        builtin_sysroot = "/dt11",
    )

cc_toolchain_config = rule(
    implementation = _impl,
    provides = [CcToolchainConfigInfo],
)
