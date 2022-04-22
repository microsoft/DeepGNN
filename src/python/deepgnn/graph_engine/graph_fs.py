# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import glob
import shutil
from azure.datalake.store import core, lib
from deepgnn.graph_engine._adl_reader import AdlCredentialParser
from pathlib import Path
from urllib.parse import urlparse


class GraphDataFS:
    def exists(self, file):
        raise NotImplementedError

    def mtime(self, file):
        raise NotImplementedError

    def touch(self, file):
        raise NotImplementedError

    def length(self, file):
        raise NotImplementedError

    def mkdir(self, path):
        raise NotImplementedError

    def rm(self, path):
        raise NotImplementedError

    def glob(self, pattern):
        raise NotImplementedError

    def open(self, file, mode):
        raise NotImplementedError


class LocalGraphDataFS(GraphDataFS):
    def __init__(self):
        pass

    def exists(self, file):
        return os.path.isfile(file)

    def mtime(self, file):
        return str(os.stat(file).st_mtime_ns)

    def touch(self, file):
        Path(file).touch()

    def mkdir(self, path):
        os.makedirs(path, exist_ok=True)

    def rm(self, path):
        shutil.rmtree(path)

    def glob(self, pattern):
        return glob.glob(pattern)

    def open(self, file, mode):
        return open(file, mode)

    def length(self, file):
        return os.stat(file).st_size


class AdlsGraphDataFS(GraphDataFS):
    def __init__(self, store_name, config=None):
        adl_config = AdlCredentialParser.read_credentials(config=config)

        self.tenantId = adl_config["TENANT_ID"]
        self.clientSecret = adl_config["CLIENT_SECRET"]
        self.clientId = adl_config["CLIENT_ID"]

        principal_token = lib.auth(
            tenant_id=self.tenantId,
            client_secret=self.clientSecret,
            client_id=self.clientId,
        )

        self.adl = core.AzureDLFileSystem(token=principal_token, store_name=store_name)

    def touch(self, file):
        # use try catch to catch the PermissionError when touching
        # ADL file in multi-threading.
        try:
            self.adl.touch(file)
        except:
            pass

    def glob(self, pattern):
        return self.adl.glob(pattern)

    def mkdir(self, path):
        self.adl.mkdir(path)

    def exists(self, file):
        return self.adl.exists(file)

    def rm(self, path):
        self.adl.remove(path, recursive=True)

    def mtime(self, file):
        return str(self.adl.stat(file)["modificationTime"])

    def open(self, file, mode):
        return self.adl.open(file, mode)

    def length(self, file):
        return self.adl.stat(file)["length"]


def parse_fs(data_dir, config: str = ""):
    """Get different FS based on the different protocol.
    Return:
        (FS, hostname, path)
    """
    if data_dir.startswith("adl:"):
        o = urlparse(data_dir)
        return (
            AdlsGraphDataFS(store_name=o.hostname.split(".")[0], config=config),
            o.hostname,
            o.path,
        )
    else:
        return LocalGraphDataFS(), "", data_dir
