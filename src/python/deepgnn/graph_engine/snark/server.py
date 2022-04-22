# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Stanalone graph engine server."""
from datetime import datetime
from ctypes import POINTER, Structure, byref, c_char_p, c_size_t, c_uint32, c_int32
from typing import Any, Dict, List

from deepgnn.graph_engine.snark._lib import _get_c_lib
from deepgnn.graph_engine.snark._downloader import download_graph_data, GraphPath
from deepgnn.graph_engine.snark.client import PartitionStorageType
from deepgnn.graph_engine.snark.meta import _set_hadoop_classpath


class _SERVER(Structure):
    _fields_: List[Any] = []


class _ErrCallback:
    def __init__(self, method: str):
        self.method = method

    # We have to use mypy ignore, to reuse this callable object across
    # all C function call because they have different signatures.
    def __call__(self, result, func, arguments):
        if result != 0:
            raise Exception(f"Failed to {self.method}")


class Server:
    """GRPC based graph engine server."""

    def __init__(
        self,
        data_path: str,
        partitions: List[int],
        hostname: str,
        ssl_config: Dict[str, str] = None,
        storage_type: PartitionStorageType = PartitionStorageType.memory,
        config_path: str = "",
        stream: bool = False,
    ):
        """Create server and start it.

        Args:
            path (str): location of graph binary files. If given hdfs:// or adl:// path see config_path and stream parameters,
                adl://name.azuredatalakestore.net/path/to
                hdfs://localhost:9000/path/to
            partitions (List[int]): which partitions to load.
            hostname (str): address to use for serving.
            ssl_config (Dict[str, str], optional): SSL configs for encrypted connections with clients if needed. Defaults to None.
            storage_type (PartitionStorageType, default=memory): What type of feature / index storage to use in GE.
            config_path (str, optional): Path to folder with configuration files.
            stream (bool, default=False): If remote path is given: by default, download files first then load,
                if stream = True and libhdfs present, stream data directly to memory.
        """
        if (
            data_path.startswith("hdfs://")
            or data_path.startswith("adl://")
            or data_path.startswith("file:///")
        ):
            _set_hadoop_classpath(config_path)

        self.seed = datetime.now()
        self.s_ = _SERVER()
        self.lib = _get_c_lib()
        self._data_path: GraphPath = (
            GraphPath(data_path)
            if stream
            else download_graph_data(data_path, partitions)
        )

        self.lib.StartServer.argtypes = [
            POINTER(_SERVER),
            c_size_t,
            POINTER(c_uint32),
            c_char_p,
            c_char_p,
            c_char_p,
            c_char_p,
            c_char_p,
            c_int32,
            c_char_p,
        ]

        self.lib.StartServer.errcheck = _ErrCallback("start server")  # type: ignore
        self.lib.StartServer.restype = c_int32
        PartitionArray = c_uint32 * len(partitions)
        partitions_array = PartitionArray(*partitions)
        ssl_key = None
        ssl_cert = None
        ssl_root = None
        if ssl_config is not None:
            ssl_key = c_char_p(bytes(ssl_config["ssl_key"], "utf-8"))
            ssl_cert = c_char_p(bytes(ssl_config["ssl_cert"], "utf-8"))
            ssl_root = c_char_p(bytes(ssl_config["ssl_root"], "utf-8"))

        self.lib.StartServer(
            byref(self.s_),
            len(partitions),
            partitions_array,
            c_char_p(bytes(self._data_path.name, "utf-8")),
            c_char_p(bytes(hostname, "utf-8")),
            ssl_key,
            ssl_cert,
            ssl_root,
            c_int32(storage_type),
            c_char_p(bytes(config_path, "utf-8")),
        )

    def reset(self):
        """Reset server and stop serving."""
        self.lib.ResetServer.argtypes = [POINTER(_SERVER)]
        self.lib.ResetServer.restype = c_int32
        self.lib.ResetServer.errcheck = _ErrCallback("reset server")  # type: ignore
        self.lib.ResetServer(byref(self.s_))
        self._data_path.reset()


if __name__ == "__main__":
    # import here for special usage of the module.
    import argparse
    import subprocess
    import logging
    import deepgnn.graph_engine.snark.server as server

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()

    def _str2list_int(v):
        if isinstance(v, list):
            return v
        if v == "":
            return []
        return [int(x) for x in v.split(",")]

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter, allow_abbrev=False
    )
    parser.add_argument(
        "--port", type=str, required=True, help="graph engine serivce listening port."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--partitions",
        type=_str2list_int,
        required=False,
        help="graph data partitions to load.",
    )
    group.add_argument(
        "--server_group",
        type=_str2list_int,
        required=False,
        help="First argument server index, second - total number of servers, third - total number of partitions.",
    )
    parser.add_argument(
        "--data_dir", type=str, required=True, help="graph data directory."
    )
    parser.add_argument(
        "--storage_type",
        type=lambda type: PartitionStorageType[type],  # type: ignore
        default=PartitionStorageType.memory,
        choices=list(PartitionStorageType.__members__.keys())
        + list(PartitionStorageType),
        help="Partition storage backing to use, eg memory or disk.",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default="",
        help="Directory where HDFS or other config files are stored.",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        default=False,
        help="If ADL data path, stream directly to memory or download to disk first.",
    )

    args, _ = parser.parse_known_args()
    if args.server_group is not None:
        assert (
            len(args.server_group) == 3
        ), "Expect 3 items: server_index, number of server, number of partitions"
        partitions = list(
            range(args.server_group[0], args.server_group[2], args.server_group[1])
        )
    else:
        partitions = args.partitions

    s = server.Server(
        args.data_dir,
        partitions,
        f"0.0.0.0:{args.port}",
        storage_type=args.storage_type,
        config_path=args.config_path,
        stream=args.stream,
    )
    logger.info("Server started...")
    try:
        subprocess.check_call("sleep infinity", shell=True)
    except KeyboardInterrupt:
        logger.info("Shutting down.")
    s.reset()
