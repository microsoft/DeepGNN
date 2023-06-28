# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""MOOC dataset."""

import argparse
import os
import csv
import zipfile
import tempfile
import operator
import json
from typing import Optional, Tuple, List, Dict

import numpy as np

from deepgnn.graph_engine.snark.meta import BINARY_DATA_VERSION
from deepgnn.graph_engine.snark.converter.writers import BinaryWriter
from deepgnn.graph_engine.data.data_util import download_file
from deepgnn.graph_engine.snark.local import Client


class MOOC(Client):
    """Temporal MOOC dataset.

    Args:
        Client (_type_): _description_
    """

    def __init__(self, output_dir: Optional[str] = None):
        """Initialize LastFM dataset."""

        self.url = "https://zenodo.org/record/7213796/files/mooc.zip"
        self.GRAPH_NAME = "mooc"
        # nodes with id < 10000 belong to users and nodes with id >= 10000 belong to items
        self._node_split = 10000

        self.output_dir = output_dir
        if self.output_dir is None:
            self.output_dir = os.path.join(tempfile.gettempdir(), self.GRAPH_NAME)

        self.writer = BinaryWriter(self.output_dir, suffix=0, watermark=0)
        self._build_graph()
        self._build_meta()
        super().__init__(path=self.output_dir, partitions=[0])

    def data_dir(self):
        """Graph location on disk."""
        return self.output_dir

    def _serialize_temporal_features(
        self, item: list[Tuple[int, int, np.array]]
    ) -> np.ndarray:
        interval_count = len(item)
        metadata_size = 2 * interval_count + 1
        data = np.zeros(4 * (2 * metadata_size + 1), dtype=np.byte)
        original_len = data.size
        count = data[0:4].view(dtype=np.int32)
        count[0] = interval_count
        temporal_metadata = data[4:].view(dtype=np.int64)
        for idx, it in enumerate(item):
            temporal_metadata[idx] = int(it[0])
            temporal_metadata[idx + interval_count] = data.size
            vals = it[2].view(dtype=np.byte)
            data = np.concatenate((data, vals))
            temporal_metadata = data[4:original_len].view(dtype=np.int64)

        temporal_metadata[metadata_size - 1] = data.size
        data = np.concatenate((data, np.zeros(8, dtype=np.byte)))
        return [data]

    def _edge_iterator(self, file):
        reader = sorted(
            csv.reader(file, delimiter=",", quoting=csv.QUOTE_NONNUMERIC),
            key=lambda t: (t[0], t[1], t[2]),
        )
        curr_src = -1
        curr_dst = -1
        features: list[Tuple[int, int, np.array]] = []

        # a single row is of the form [src, dst, timestamp, label, features]
        # it represents an edge from src to dst with a single edge feature
        # this means we need to aggregate all features manually before emitting the edge
        for row in reader:
            node_id = int(row[0])
            if curr_dst == -1:
                curr_dst = int(row[1])
            if node_id != curr_src:
                yield node_id, -1, 0, 1.0, 0, -1, []
                curr_dst = int(row[1])
                curr_src = node_id
            if int(row[1]) != curr_dst and len(features) > 0:
                yield curr_src, curr_dst, 0, 1.0, features[0][
                    0
                ], -1, self._serialize_temporal_features(features)
                features = []
                curr_dst = int(row[1])
            features.append((int(row[2]), -1, np.array(row[4:], dtype=np.float32)))
            if len(features) > 1:
                features[-2] = (features[-2][0], features[-1][0], features[-2][2])

        yield curr_src, int(row[1]), 0, 1.0, features[0][
            0
        ], -1, self._serialize_temporal_features(features)

    def _build_graph(self) -> str:
        raw_data_dir = os.path.join(self.output_dir, "raw")
        download_file(self.url, raw_data_dir, f"{self.GRAPH_NAME}.zip")
        fname = os.path.join(raw_data_dir, f"{self.GRAPH_NAME}.zip")
        with zipfile.ZipFile(fname) as z:
            z.extractall(raw_data_dir)
        with open(os.path.join(raw_data_dir, "mooc", "mooc.csv")) as csvfile:
            next(csvfile)  # skip header
            self.writer.add(self._edge_iterator(csvfile))
            self.writer.close()

    def _build_meta(self):
        mjson = {
            "binary_data_version": BINARY_DATA_VERSION,
            "node_count": self.writer.node_count,
            "edge_count": self.writer.edge_count,
            "node_type_count": 1,
            "edge_type_count": 1,
            "node_feature_count": 0,
            "edge_feature_count": 1,
        }

        edge_count_per_type = [self.writer.edge_count]
        node_count_per_type = [self.writer.node_count]
        mjson["partitions"] = {}

        mjson["partitions"]["0"] = {
            "node_weight": [self.writer.node_count],
            "edge_weight": [self.writer.edge_count],
        }

        mjson["node_count_per_type"] = node_count_per_type
        mjson["edge_count_per_type"] = edge_count_per_type
        mjson["watermark"] = 1

        with open(f"{self.output_dir}/meta.json", "w") as file:
            file.write(json.dumps(mjson))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=f"{tempfile.gettempdir()}/mooc", type=str)
    args = parser.parse_args()
    MOOC(args.data_dir)
