# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Decoders wihch is used to decode a line of text into node object."""
import abc
import json
import csv
from typing import TypeVar, Iterator, Tuple
import numpy as np


class Decoder(abc.ABC):
    """Interface to convert one line of text into node object."""

    # For converting json strings to numpy types.
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
    def decode(self, line: str) -> Iterator[Tuple[int, int, int, float, list]]:
        """Decode the line of text.

        This is a generator that yields a node then its outgoing edges in order.
        Yield format is (node_id/src, -1/dst, type, weight, features).
        Features being a list of dense features as ndarrays and sparse features as 2 tuples, coordinates and values.
        """


DecoderType = TypeVar("DecoderType", bound=Decoder)


class JsonDecoder(Decoder):
    """Convert the text line into node object using json.

    Json format:
        {
            "node_id": 0,
            "node_type": 1,
            "node_weight": 1,
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
        """From item, pull all value dicts {idx: value} and order the values by idx."""
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
                        np.array(
                            value["values"],
                            dtype=self.convert_map[key.replace("sparse_", "")],
                        ),
                    )
                elif key != "binary_feature":
                    value = np.array(value, dtype=self.convert_map[key])
                ret_list[idx] = value

        return ret_list

    def decode(self, line: str) -> Iterator[Tuple[int, int, int, float, list]]:
        """Use json package to convert the json text line into a node and edge iterator.

        This is a generator that yields a node then its outgoing edges in order.
        Yield format is (node_id/src, -1/dst, type, weight, features).
        Features being a list of dense features as ndarrays and sparse features as 2 tuples, coordinates and values.
        """
        data = json.loads(line)
        yield data["node_id"], -1, data["node_type"], data[
            "node_weight"
        ], self._pull_features(data)
        for edge in sorted(data["edge"], key=lambda x: x["edge_type"]):
            yield edge["src_id"], edge["dst_id"], edge["edge_type"], edge[
                "weight"
            ], self._pull_features(edge)


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

    def decode(self, line: str) -> Iterator[Tuple[int, int, int, float, list]]:
        """Decode tsv based text line into a node and edge iterator.

        This is a generator that yields a node then its outgoing edges in order.
        Yield format is (node_id/src, -1/dst, type, weight, features).
        Features being a list of dense features as ndarrays and sparse features as 2 tuples, coordinates and values.
        """
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
        yield node_id, -1, node_type, node_weight, self._parse_feature_string(
            columns[3]
        )

        if not columns[4]:
            return

        neighbors = columns[4].split("|")
        for n in sorted(neighbors, key=lambda x: int(x.split(",")[1])):
            neighbor_columns = n.split(",")
            # make sure the destination id is valid
            assert len(neighbor_columns) > 0 and len(neighbor_columns[0]) > 0
            dst_id = int(neighbor_columns[0])

            for i in range(TsvDecoder.NEIGHBOR_COLUMN_COUNT - len(neighbor_columns)):
                neighbor_columns.append("")

            dst_type = int(neighbor_columns[1]) if neighbor_columns[1] else 0
            dst_weight = float(neighbor_columns[2]) if neighbor_columns[2] else 0.0
            yield node_id, dst_id, dst_type, dst_weight, self._parse_feature_string(
                neighbor_columns[3]
            )
