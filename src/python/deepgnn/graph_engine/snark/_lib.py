# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import platform
import functools

if platform.system() == "Windows":
    from ctypes import WinDLL  # type: ignore
else:
    from ctypes import CDLL  # type: ignore

# TODO: Use .dylib extension instead of .so on MacOS.
#       Related issue: https://github.com/bazelbuild/bazel/issues/11082
_LIB_FILE_NAME = "libwrapper.so"
if platform.system() == "Windows":
    _LIB_FILE_NAME = "wrapper.dll"

_LIB_PATH = os.path.join(os.path.dirname(__file__), _LIB_FILE_NAME)

# Use environment variables to load library with multiprocessing module
_SNARK_LIB_PATH_ENV_KEY = "SNARK_LIB_PATH"


# Use lru_cache for to load library only once in thread safe mode.
@functools.lru_cache(maxsize=1)
def _get_c_lib():
    global _LIB_PATH

    if _SNARK_LIB_PATH_ENV_KEY in os.environ:
        _LIB_PATH = os.environ[_SNARK_LIB_PATH_ENV_KEY]

    if platform.system() == "Windows":
        lib = WinDLL(_LIB_PATH)  # type: ignore
    else:
        lib = CDLL(_LIB_PATH)  # type: ignore
    return lib
