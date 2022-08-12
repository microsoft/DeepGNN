# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import doctest
import os
import platform
import sys
import deepgnn

LIB_NAME = "libwrapper.so"
if platform.system() == "Windows":
    LIB_NAME = "wrapper.dll"

if __name__ == "__main__":
    os.environ["SNARK_LIB_PATH"] = os.path.join(
        os.getcwd(), "src", "cc", "lib", LIB_NAME
    )
    res = doctest.testfile(sys.argv[1], report=True, optionflags=doctest.ELLIPSIS)
    assert res.failed == 0
