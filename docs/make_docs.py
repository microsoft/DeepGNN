# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Helper class to generate docs with sphinx."""
import sphinx.cmd.build as build

build.build_main(argv=["-n", "-b", "html", "docs", "_build"])
