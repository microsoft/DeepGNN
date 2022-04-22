# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Decoders wihch is used to decode a line of text into node object."""
import abc
import json
import logging
import csv
from enum import Enum
from typing import Any, Dict

logger = logging.getLogger()


class DecoderType(Enum):
    """Decoder types supported by converter."""

    JSON = "json"
    TSV = "tsv"

    def __str__(self):
        """Convert instance to string."""
        return self.value


class Decoder(abc.ABC):
    """Interface to convert one line of text into node object."""

    @abc.abstractmethod
    def decode(self, line: str):
        """Decode the line into a "node" object."""


class JsonDecoder(Decoder):
    """Convert the text line into node object using json.

    Json format:
        {
            "node_id": 0,
            "node_type": 1,
            "node_weight": 1,
            "neighbor": {"0": {}, "1": {"5": 1}},
            "uint64_feature": {},
            "float_feature": {"0": [1], "1": [-0.03, -0.04]},
            "binary_feature": {},
            "edge": [
                {
                    "src_id": 0,
                    "dst_id": 5,
                    "edge_type": 1,
                    "weight": 1,
                    "uint64_feature": {},
                    "float_feature": {"1": [3, 4]},
                    "binary_feature": {},
                }
            ]
        }
    """

    def __init__(self):
        """Initialize the JsonDecoder."""
        super().__init__()

    def decode(self, line: str):
        """Use json package to convert the json text line into node object."""
        return json.loads(line)


class TsvDecoder(Decoder):
    """TSV Decoder convert the tsv baesd text line into node object."""

    COLUMN_COUNT = 5
    NEIGHBOR_COLUMN_COUNT = 4

    """Convert the tsv based text line into node object.
    TSV format:
        node_id:        (int)
                        e.g. 1
        node_type:      (int)
                        e.g. 0
        node_weight:    (float)
                        e.g. 0.1
        node_features:  (feature_type: feature_content;feature_type: feature_content)
                        feature_type: i(nt)8/16/32/64, u(int)8/16/32/64, f(loat)16, f(loat), d(ouble), b(inary)
                        e.g. f:0.1 0.2;b:stringfeatures;i:1 2 3;i16: 1 2 3;u32: 1 2 3
        neighbors:      (dst_id,edge_type,edge_weight,edge_label,edge_features|dst_id,edge_type,edge_weight,edge_label,edge_features|...)
                        e.g. 2, 0, 0.3, 1, 0.1 0.2:stringfeatures:1 2 3|3, 1, 0.4, 2, 0.1 0.2:stringfeatures:4 5 6|...

    Example:
            | node_id | node_type | node_weight | node_features                | neighbors                                  |
            | --------|-----------|-------------|------------------------------|--------------------------------------------|
            | 1       | 0         | 0.1         | f:0.1 0.2;b:str_feat;i:1 2 3 | 2, 0, 0.3, 1, f:0.1 0.2;b:str_feat;i:1 2 3 |
    """

    def __init__(self):
        """Initialize TSV decoder."""
        super().__init__()

    def _parse_feature_string(self, raw_feature):
        index = 0
        feature_map: Dict[str, Any] = dict()
        node_features = raw_feature.split(";") if raw_feature else []
        for feature in node_features:
            val = feature.split(":")
            if len(val) == 2 and val[0] and val[1]:
                if val[0][0] == "i":
                    key = f"int{val[0][1:]}_feature"
                    if key not in feature_map:
                        feature_map[key] = dict()
                    feature_map[key][str(index)] = [int(i) for i in val[1].split(" ")]
                elif val[0][0] == "u":
                    key = f"uint{val[0][1:]}_feature"
                    if key not in feature_map:
                        feature_map[key] = dict()
                    feature_map[key][str(index)] = [int(i) for i in val[1].split(" ")]
                elif val[0][0] == "f":
                    key = f"float{val[0][1:]}_feature"
                    if key not in feature_map:
                        feature_map[key] = dict()
                    feature_map[key][str(index)] = [float(i) for i in val[1].split(" ")]
                elif val[0][0] == "d":
                    key = "double_feature"
                    if key not in feature_map:
                        feature_map[key] = dict()
                    feature_map[key][str(index)] = [float(i) for i in val[1].split(" ")]
                elif val[0][0] == "b":
                    key = "binary_feature"
                    feature_map[key] = dict()
                    feature_map[key][str(index)] = val[1]

            index = index + 1

        return feature_map

    def decode(self, line: str):
        """Decode tsv based text line into node object."""
        assert line is not None and len(line) > 0

        columns = next(csv.reader([line], delimiter="\t"))
        # fill the missing columns with default empty string, so we
        # don't need to check if the column exists.
        for i in range(TsvDecoder.COLUMN_COUNT - len(columns)):
            columns.append("")

        node: Dict[str, Any] = dict()

        # make sure the node id is valid
        assert columns[0] is not None and len(columns[0]) > 0
        node["node_id"] = int(columns[0])
        node["node_type"] = int(columns[1]) if columns[1] else 0
        node["node_weight"] = float(columns[2]) if columns[2] else 0.0

        # index 1 of the float feature is node float feature.
        feature_map = self._parse_feature_string(columns[3])
        node.update(feature_map)

        node["neighbor"] = dict()
        node["edge"] = []
        if columns[4]:
            neighbors = columns[4].split("|")
            for n in neighbors:
                neighbor_columns = n.split(",")
                # make sure the destination id is valid
                assert len(neighbor_columns) > 0 and len(neighbor_columns[0]) > 0
                dst_id = int(neighbor_columns[0])

                for i in range(
                    TsvDecoder.NEIGHBOR_COLUMN_COUNT - len(neighbor_columns)
                ):
                    neighbor_columns.append("")

                dst_type = int(neighbor_columns[1]) if neighbor_columns[1] else 0
                dst_weight = float(neighbor_columns[2]) if neighbor_columns[2] else 0.0

                if dst_type not in node["neighbor"]:
                    node["neighbor"][str(dst_type)] = dict()

                node["neighbor"][str(dst_type)][str(dst_id)] = dst_weight

                edge_info = dict()
                edge_info["src_id"] = node["node_id"]
                edge_info["dst_id"] = dst_id
                edge_info["edge_type"] = dst_type
                edge_info["weight"] = dst_weight

                feature_map = self._parse_feature_string(neighbor_columns[3])
                edge_info.update(feature_map)

                node["edge"].append(edge_info)

        return node
