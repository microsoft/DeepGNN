# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import sys
import pytest
import platform

LIB_NAME = "libwrapper.so"
if platform.system() == "Windows":
    LIB_NAME = "wrapper.dll"


def test_notebooks():
    os.environ["SNARK_LIB_PATH"] = f"{os.getcwd()}/src/cc/lib/{LIB_NAME}"
    ret = os.system("python -m pytest -s --nbval-lax -o testpaths=docs/tutorials/tf/")
    assert ret == 0

    ret = os.system(
        "python -m pytest -s --nbval-lax -o testpaths=docs/tutorials/pytorch/"
    )
    assert ret == 0

    ret = os.system(
        "python -m pytest -s --nbval-lax -o testpaths=docs/tutorials/other/"
    )
    assert ret == 0


if __name__ == "__main__":
    sys.exit(
        pytest.main(
            [__file__, "--junitxml", os.environ["XML_OUTPUT_FILE"], *sys.argv[1:]]
        )
    )
