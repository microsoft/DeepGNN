# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import doctest
import os
import platform
import sys
import deepgnn

if __name__ == "__main__":
    res = doctest.testfile(sys.argv[1], report=True, optionflags=doctest.ELLIPSIS)
    assert res.failed == 0
