# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
from typing import List
from enum import Enum
from deepgnn.graph_engine.snark.converter.options import ConverterOptions
from deepgnn.graph_engine.snark.client import PartitionStorageType


class GraphType(Enum):
    LOCAL = "local"
    REMOTE = "remote"

    def __str__(self):
        return self.value


class BackendOptions:
    def __init__(self, params: argparse.Namespace):
        self.backend = None
        self.data_dir = ""
        # local GE only for local debugging.
        self.graph_type = GraphType.REMOTE
        self.model_dir = ""
        # Snark parameters
        self.ge_start_timeout = 30
        self.num_ge = 0
        self.partitions: List[int] = []
        self.servers: List[str] = []
        self.server_idx = -1
        self.client_rank = -1
        self.skip_ge_start = False
        self.sync_dir = ""
        self.enable_ssl = False
        self.ssl_cert = ""
        self.storage_type = PartitionStorageType.memory
        self.config_path = ""
        self.stream = False

        # sometimes user need to implement their own backend, using this custom
        # field, user can start graph engine using their own code.
        self.custom_backendclass = None

        for arg in vars(params):
            setattr(self, arg, getattr(params, arg))

        self.converter_options = ConverterOptions(params)
