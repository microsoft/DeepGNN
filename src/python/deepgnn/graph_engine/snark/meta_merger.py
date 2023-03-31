# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Merge meta data files generated by distributed converters."""
import os
from typing import List
import json

import fsspec

from deepgnn.graph_engine._base import get_fs

# Content of sample meta.json

# Fields            Description             Action
# ====================================================
# v1                binary version          equal
# 2708              node_count              additive
# 10556             edge_count              additive
# 4                 node_type_num           equal
# 1                 edge_type_num           equal
# 2                 node_xxx_feature_num    equal
# 0                 edge_xxx_feature_num    equal
# 1                 partitions count        equal
# 0                 Partition id            equal
# 140.0             node weight 1           additive
# 500.0             node weight 2           additive
# 1000.0            node weight 3           additive
# 1068.0            node weight 4           additive
# 10556.0           edge weight 1           additive
# 140               node count type 1       additive
# 500               node count type 2       additive
# 1000              node count type 3       additive
# 1068              node count type 4       additive
# 10556             edge count type 1       additive


class Meta:
    """Merge meta data files generated by distributed converters."""

    def __init__(self, fs: fsspec.AbstractFileSystem, path: str) -> None:
        """Initialize Meta merger.

        fs: abstract fsspec file system.
        path: path of the meta file.
        """
        self.fs = fs
        self.path = path

        with self.fs.open(self.path, "r") as file:
            meta = json.load(file)  # type: ignore

        self.version = meta["binary_data_version"]
        if self.version[0] != "v":
            raise RuntimeError(
                "Meta file should contain binary_data_version, please regenerate binary data"
            )

        self.node_count = meta["node_count"]
        self.edge_count = meta["edge_count"]
        self.node_type_num = meta["node_type_num"]
        self.edge_type_num = meta["edge_type_num"]
        self.node_feature_count = meta["node_feature_num"]
        self.edge_feature_count = meta["edge_feature_num"]
        self.partition_count = len(meta["partitions"])
        self.node_weights = [[0.0] * self.node_type_num] * self.partition_count
        self.edge_weights = [[0.0] * self.edge_type_num] * self.partition_count
        for j, id in enumerate(meta["partitions"]):
            for i in range(self.node_type_num):
                self.node_weights[j][i] += meta["partitions"][id]["node_weight"][i]
            for i in range(self.edge_type_num):
                self.edge_weights[j][i] += meta["partitions"][id]["edge_weight"][i]

        self.node_count_per_type = meta["node_count_per_type"]
        self.edge_count_per_type = meta["edge_count_per_type"]

    def get_partition(self, offset):
        """Return the sub content of meta info."""
        return {
            f"{offset + i}": {
                "node_weight": self.node_weights[i],
                "edge_weight": self.edge_weights[i],
            }
            for i in range(self.partition_count)
        }, offset + self.partition_count


def merge(fs, output_dir: str, meta_list: List[Meta]):
    """Merge meta files."""
    assert all(
        map(lambda x: x.version == meta_list[0].version, meta_list)
    ), "All partitions must have the same binary version"
    content = {
        "binary_data_version": meta_list[0].version,
        "node_count": sum([i.node_count for i in meta_list]),
        "edge_count": sum([i.edge_count for i in meta_list]),
        "node_type_num": meta_list[0].node_type_num,
        "edge_type_num": meta_list[0].edge_type_num,
        "node_feature_num": int(meta_list[0].node_feature_count),
        "edge_feature_num": int(meta_list[0].edge_feature_count),
    }
    offset = 0
    content["partitions"] = {}
    for i in meta_list:
        data, offset = i.get_partition(offset)
        content["partitions"].update(data)

    content["node_count_per_type"] = [
        sum([i.node_count_per_type[count] for i in meta_list])
        for count in range(int(meta_list[0].node_type_num))
    ]
    content["edge_count_per_type"] = [
        sum([i.edge_count_per_type[count] for i in meta_list])
        for count in range(int(meta_list[0].edge_type_num))
    ]

    with fs.open("{}/meta.json".format(output_dir), "w") as file:
        file.write(json.dumps(content))


def merge_metadata_files(meta_path: str):
    """Entrance of the merger."""
    fs, _ = get_fs(meta_path)

    total_files = [
        f if isinstance(f, str) else f["name"]
        for f in fs.glob(os.path.join(meta_path, "meta_*"))
    ]

    if len(total_files) == 0:
        return

    meta_list = [Meta(fs, f) for f in total_files]
    merge(fs, meta_path, meta_list)


if __name__ == "__main__":
    # import here for special usage of the module.
    import argparse

    parser = argparse.ArgumentParser(description="Merge all meta.json.")
    parser.add_argument("-d", "--data_dir", help="meta.json folder path")
    args = parser.parse_args()
    merge_metadata_files(args.data_dir)
