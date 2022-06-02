# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Decoders wihch is used to decode a line of text into node object."""
import abc
import dataclasses
import json
import logging
import csv
from enum import Enum
from typing import Any, Dict
import numpy as np

logger = logging.getLogger()


class DecoderType(Enum):
    """Decoder types supported by converter."""

    JSON = "json"
    TSV = "tsv"
    LINEAR = "linear"

    def __str__(self):
        """Convert instance to string."""
        return self.value


class Decoder(abc.ABC):
    """Interface to convert one line of text into node object."""
    convert_map = {  # TODO all + sparse
        "float_feature": np.float32,
        "uint64_feature": np.uint64,
    }

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

    def _pull_features(self, item: dict) -> list:
        """From item, pull all value dicts {idx: value} and order the values by idx"""
        ret_list = []  # type: ignore
        curr = 0
        for key, values in item.items():
            if values is None or "feature" not in key:
                continue

            for idx, value in values.items():
                idx = int(idx)
                while curr <= idx:
                    ret_list.append(None)
                    curr += 1
                if key != "binary_feature":
                    value = np.array(value, dtype=self.convert_map[key])
                ret_list[idx] = value

        return ret_list

    def decode(self, line: str):
        """Use json package to convert the json text line into node object."""
        data = json.loads(line)
        yield -1, data["node_id"], data["node_type"], data["node_weight"], self._pull_features(data)
        for edge in data["edge"]:
            yield edge["src_id"], edge["dst_id"], edge["edge_type"], edge["weight"], self._pull_features(edge)


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
        feature_map = []
        node_features = raw_feature.split(";") if raw_feature else []
        for feature in node_features:
            val = feature.split(":")
            if len(val) != 2 or not val[0] or not val[1]:
                continue

            if val[0][0] == "i":
                key = f"int{val[0][1:]}_feature"
            elif val[0][0] == "u":
                key = f"uint{val[0][1:]}_feature"
            elif val[0][0] == "f":
                key = f"float{val[0][1:]}_feature"
            elif val[0][0] == "d":
                key = "double_feature"
            elif val[0][0] == "b":
                key = "binary_feature"

            if key != "binary_feature":
                value = np.array(val[1].split(" "), dtype=self.convert_map[key])
            else:
                value = val[1]

            feature_map.append(value)

        return feature_map

    def decode(self, line: str):
        """Decode tsv based text line into node object."""
        assert line is not None and len(line) > 0
        columns = next(csv.reader([line], delimiter="\t"))
        # fill the missing columns with default empty string, so we
        # don't need to check if the column exists.
        for i in range(TsvDecoder.COLUMN_COUNT - len(columns)):
            columns.append("")

        # make sure the node id is valid
        assert columns[0] is not None and len(columns[0]) > 0
        node_id = int(columns[0])
        node_type = int(columns[1]) if columns[1] else 0
        node_weight = float(columns[2]) if columns[2] else 0.0
        yield -1, node_id, node_type, node_weight, self._parse_feature_string(columns[3])

        if not columns[4]:
            return

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
            yield node_id, dst_id, dst_type, dst_weight, self._pull_features(self._parse_feature_string(neighbor_columns[3]))


class LinearDecoder(Decoder):
    """Convert the text line into node object.
    Linear format:
        node_id -1 node_type node_weight feature_dict(no spaces)
        src_id dst_id edge_type edge_weight feature_dict(no spaces)
        ...
        sorted by src with node before edges.
    Linear format example:
        0 -1 1 1 {"float_feature":{"0":[1],"1":[-0.03,-0.04]}}
        0 5 1 1 {"1":[3,4]}
    """

    def __init__(self):
        """Initialize the Decoder."""
        super().__init__()

    def decode(self, line: str):
        """Use json package to convert the json text line into node object."""
        #for line in lines:
        data = line.split()

        idx = 0
        while True:
            try:
                src, dst, typ, weight = data[idx:idx+4]
            except ValueError:
                break
            idx += 4
            features = []
            while True:
                try:
                    key = data[idx]
                    try:
                        int(key)
                        break
                    except Exception as e:
                        pass
                    length = int(data[idx+1])
                except IndexError:
                    break
                idx += 2
                if not length:
                    value = None
                elif length == 1 and key == "binary_feature":
                    value = data[idx]
                else:
                    value = np.array(data[idx:idx+length], dtype=self.convert_map[key])
                features.append(value)
                idx += length
            yield int(src), int(dst), int(typ), float(weight), features


def _dump_features(features: dict) -> str:
    """Serialize features for linear format."""
    output = []
    for key, values in features.items():
        if not isinstance(values, dict) or key == "neighbor":
            continue

        for idx, value in values.items():
            if key == "binary_feature":
                v = str(value)
                length = 1
            else:
                v = np.array(value).dumps()  #" ".join(map(str, value))
                length = v.size()
            output.append(f"{key} {length} {v}")
    
    return " ".join(output)


def json_to_linear(filename_in, filename_out):
    """Convert graph.json to graph.linear."""
    file_in = open(filename_in, "r")
    file_out = open(filename_out, "w")

    for line in file_in.readlines():
        node = json.loads(line)

        file_out.write(
            f'-1 {node["node_id"]} {node["node_type"]} {node["node_weight"]} {_dump_features(node)}'
        )

        #edge_list = sorted(
        #    node["edge"], key=lambda x: (int(x["edge_type"]), int(x["dst_id"]))
        #)
        for edge in node["edge"]:
            file_out.write(
                f' {edge["src_id"]} {edge["dst_id"]} {edge["edge_type"]} {edge["weight"]} {_dump_features(edge)}'
            )
        file_out.write('\n')

    file_in.close()
    file_out.close()
