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


class Decoder(abc.ABC):
    """Interface to convert one line of text into node object."""
    convert_map = {
        "double_feature": np.float64,
        "float_feature": np.float32,
        "float64_feature": np.float64,
        "float32_feature": np.float32,
        "float16_feature": np.float16,
        "uint64_feature": np.uint64,
        "int64_feature": np.int64,
        "uint32_feature": np.uint32,
        "int32_feature": np.int32,
        "uint16_feature": np.uint16,
        "int16_feature": np.int16,
        "uint8_feature": np.uint8,
        "int8_feature": np.int8,
    }

    @abc.abstractmethod
    def decode(self, line: str):
        """Decode the line into a "node" object."""


class LinearDecoder(Decoder):
    """Convert the text line into node object.
    Linear format:
        {node info} {edge_1_info} {edge_2_info} ...
        node_info: -1 node_id node_type node_weight {features}
        edge_info: src dst edge_type edge_weight {features}
        features[dense]: dtype_name length v1 v2 ... dtype_name2 length2 v1 v2 ...
        features[sparse]: dtype_name coords.size,values.size c1 c2 ... v1 v2 ...
        features[sparse]: dtype_name coords.shape[0],coords.shape[1],values.size c1 c2 ... v1 v2

    Linear Format Example
        -1 0 1 .5 int32 3 1 1 1 float32 2 1.1 1.1 0 1 0 .5 uint8 2,3 0 4 1 1 1
        -1 1 1 .5 int32 3 1 1 1 float32 2 1.1 1.1 1 0 0 .5 uint8 2,3 0 4 1 1 1
    """

    def __init__(self):
        """Initialize the Decoder."""
        super().__init__()

        self.node_type, self.node_weight, self.node_feature_types, self.node_feature_lens = None, None, [], []
        self.edge_type, self.edge_weight, self.edge_feature_types, self.edge_feature_lens = None, None, [], []
    
    def encode(self, node_id: int, node_type: int, node_weight: float, node_features: list, edges: list) -> str:
        """
        Convert data to a line of the linear format.

        Parameters
        ----------
        node_id: int
        node_type int
        node_weight: float
        node_features: [ndarray, ...]
        edges: [(edge_src: int, edge_dst: int, edge_type: int, edge_weight: float, edge_features[ndarray, ...]), ...]

        Return
        ------
        str Linear format version of input.
        """
        def _feature_str(features):
            def get_f(f):
                if f is None:
                    return "uint8 0"
                elif isinstance(f, str):
                    return f"binary_feature 1 {f}"
                elif isinstance(f, tuple):
                    coords, values = f
                    if len(coords.shape) == 1:
                        coordinates_str = " ".join(map(str, coords))
                        length = f"{coords.shape[0]}"
                    else:
                        coordinates_str = " ".join((" ".join(map(str, c)) for c in coords))
                        length = f"{coords.shape[0]},{coords.shape[1]}"

                    return f"{values.dtype.name} {length},{values.size} {coordinates_str} {' '.join(map(str, values))}"
                return f"{f.dtype.name} {f.size} {' '.join(map(str, f))}"

            # TODO None and string case
            return " ".join(get_f(f) for f in features)
        output = f"-1 {node_id} {node_type} {node_weight} {_feature_str(node_features)}"
        for edge_src, edge_dst, edge_type, edge_weight, edge_features in edges:
            if output[-1] != " ":
                output += " "
            output += f"{edge_src} {edge_dst} {edge_type} {edge_weight} {_feature_str(edge_features)}"
        return output

    def _parse_first_line(self, idx, data: str):
        item_type, item_weight, item_features_types, item_features_lens = None, None, [], []
        try:
            v = data[idx]
            if v.lower() != "none":
                item_type = int(v)
            idx += 1
        except (IndexError, ValueError): 
            pass
        try:
            v = data[idx]
            if v.lower() != "none":
                item_weight = float(v)
            idx += 1
        except (IndexError, ValueError):
            pass
        while True:
            try:
                item_features_type = data[idx]
                int(item_features_type)
                if item_features_type == "edge_defaults":
                    break
                if item_features_type.lower() == "none":
                    item_features_type = None
                item_features_types.append(item_features_type)
                idx += 1
            except (IndexError, ValueError): 
                break
            try:
                item_features_len = int(data[idx])
                item_features_lens.append(item_features_len)
                idx += 1
            except (IndexError, ValueError): 
                continue
        return idx, item_type, item_weight, item_features_types, item_features_lens

    def decode(self, line: str):
        """Use json package to convert the json text line into node object."""
        data = line.split()
        if not len(data):
            return []

        # (line optional)node_defaults type weight node_feature_type_1 node_feature_len_1 edge_defaults ...
        idx = 0
        # TODO only check on first node somehow
        if data[idx] == "node_defaults":
            idx, self.node_type, self.node_weight, self.node_feature_types, self.node_feature_lens = self._parse_first_line(idx + 1, data)
        if len(data) > idx and data[idx] == "edge_defaults":
            idx, self.edge_type, self.edge_weight, self.edge_feature_types, self.edge_feature_lens = self._parse_first_line(idx + 1, data)

        if idx:
            return []

        while True:
            try:
                src, dst = data[idx:idx+2]
                if idx == 0:
                    typ, weight = self.node_type, self.node_weight
                else:
                    typ, weight = self.edge_type, self.edge_weight
                idx += 2
                if typ is None:
                    typ = data[idx]
                    idx += 1
                if weight is None:
                    weight = data[idx]
                    idx += 1
            except ValueError:
                break

            features = []
            while True:
                try:
                    key = data[idx]

                    # TODO check lengthsa sa well
                    try:
                        int(key)
                        break
                    except ValueError:
                        pass
                    length = list(map(int, data[idx+1].split(",")))
                except IndexError:
                    break
                idx += 2
                if len(length) > 1:
                    coordinates_len = length[:-1]
                    coordinates_len_total = 1
                    for v in coordinates_len:
                        coordinates_len_total *= v
                    values_len = length[-1]
                    coordinates_offset = idx + coordinates_len_total

                    # TODO no sparse_binary_feature?
                    if not coordinates_len:
                        value = None
                    else:
                        value = (
                            np.array(data[idx:coordinates_offset], dtype=np.int64).reshape(coordinates_len),
                            np.array(data[coordinates_offset:coordinates_offset+values_len], dtype=key),
                        )
                    idx += coordinates_len_total + values_len
                else:
                    length = length[0]
                    if not length:
                        value = None
                    elif length == 1 and key == "binary_feature":
                        value = data[idx]
                    else:
                        value = np.array(data[idx:idx+length], dtype=key)
                    idx += length

                features.append(value)
            
            yield int(src), int(dst), int(typ), float(weight), features


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
                if key.startswith("sparse"):
                    value = (
                        np.array(value["coordinates"], dtype=np.int64),
                        np.array(value["values"], dtype=self.convert_map[key.replace("sparse_", "")]),
                    )
                elif key != "binary_feature":
                    value = np.array(value, dtype=self.convert_map[key])
                ret_list[idx] = value

        return ret_list

    def decode(self, line: str):
        """Use json package to convert the json text line into node object."""
        data = json.loads(line) if isinstance(line, str) else line
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
                feature_map.append(None)
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

            if len(val) != 2 or not val[0] or not val[1]:
                value = None
            elif key != "binary_feature":
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
            yield node_id, dst_id, dst_type, dst_weight, self._parse_feature_string(neighbor_columns[3])


def json_node_to_linear(node):
    """Convert graph.json to graph.linear."""
    gen = JsonDecoder().decode(node)
    node = next(gen)[1:]
    edges = [edge for edge in gen]
    output = LinearDecoder().encode(*node, edges)
    output += '\n'
    return output


def json_to_linear(filename_in, filename_out):
    """Convert graph.json to graph.linear."""
    file_in = open(filename_in, "r")
    file_out = open(filename_out, "w")
    for line in file_in.readlines():
        node = json.loads(line)
        file_out.write(json_node_to_linear(node))
    file_in.close()
    file_out.close()
