# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
from deepgnn import get_logger


def disable_infini_band():
    """Disable InifiniBand for communication."""
    os.environ["GLOO_SOCKET_IFNAME"] = "eth0"
    os.environ["NCCL_SOCKET_IFNAME"] = "eth0"
    os.environ["NCCL_IB_DISABLE"] = "1"
    get_logger().warn("InfiniBand(IB) has been disabled, use eth0 instead.")
