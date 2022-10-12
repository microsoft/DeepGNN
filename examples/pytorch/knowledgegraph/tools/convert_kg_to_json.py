# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Simple converter from kg to json format for deepgnn."""
import json
import sys
from typing import Dict, Any


def main(root_path: str):
    """Script entry point."""
    entities_path = root_path + "/entities.dict"
    relations_path = root_path + "/relations.dict"
    train_path = root_path + "/train.txt"

    entities_dict = {}
    relations_dict = {}

    with open(entities_path, "r") as f:
        for line in f:
            line = line.rstrip()
            cols = line.split("\t")
            entities_dict[cols[1]] = int(cols[0])

    with open(relations_path, "r") as f:
        for line in f:
            line = line.rstrip()
            cols = line.split("\t")
            relations_dict[cols[1]] = int(cols[0])

    nodes_dict: Dict[int, list] = {}
    count_ = {}
    start_ = 4
    with open(train_path, "r") as f:
        for line in f:
            line = line.rstrip()
            cols = line.split("\t")
            if cols[0] not in entities_dict:
                raise ValueError("This entity is not in the entity dict: " + cols[0])
            if cols[2] not in entities_dict:
                raise ValueError("This entity is not in the entity dict: " + cols[2])
            if cols[1] not in relations_dict:
                raise ValueError(
                    "This relation type is not in the relation dict: " + cols[1]
                )
            src_id = entities_dict[cols[0]]
            dst_id = entities_dict[cols[2]]
            rlt_id = relations_dict[cols[1]]

            if src_id not in nodes_dict:
                nodes_dict[src_id] = []
            nodes_dict[src_id].append([rlt_id, dst_id])
            # Add reverse triple for negative exampling.
            if dst_id not in nodes_dict:
                nodes_dict[dst_id] = []
            nodes_dict[dst_id].append([-rlt_id - 1, src_id])
            if (src_id, rlt_id) not in count_:
                count_[(src_id, rlt_id)] = start_
            else:
                count_[(src_id, rlt_id)] += 1

            if (dst_id, -rlt_id - 1) not in count_:
                count_[(dst_id, -rlt_id - 1)] = start_
            else:
                count_[(dst_id, -rlt_id - 1)] += 1

    nodes = []
    for k in nodes_dict.keys():
        node: Dict[str, Any] = {}
        node["node_id"] = int(k)
        node["node_type"] = 0
        node["node_weight"] = 1
        node["uint64_feature"] = {}
        node["float_feature"] = {}
        node["binary_feature"] = {}
        neighbors: Dict[int, Dict[int, int]] = {}
        edges = []

        for entity in nodes_dict[int(k)]:
            if entity[0] >= 0:
                edge_type = 0
                subsampling_weight = (
                    count_[(int(k), entity[0])] + count_[(entity[1], -entity[0] - 1)]
                )
                edge = {
                    "src_id": int(k),
                    "dst_id": entity[1],
                    "edge_type": edge_type,
                    "weight": 1,
                    "uint64_feature": {},
                    "float_feature": {},
                    "binary_feature": {},
                }
                # Add edge type and sampling weigh as uint64 features for the edge.
                edge["uint64_feature"] = {"0": [entity[0], subsampling_weight]}
                if edge_type not in neighbors:
                    neighbors[edge_type] = {}
                neighbors[edge_type][entity[1]] = 1
                edges.append(edge)
            else:
                edge_type = 1
                edge = {
                    "src_id": int(k),
                    "dst_id": entity[1],
                    "edge_type": edge_type,
                    "weight": 1,
                    "uint64_feature": {},
                    "float_feature": {},
                    "binary_feature": {},
                }
                edge["uint64_feature"] = {"0": []}
                edges.append(edge)

        node["edge"] = edges
        nodes.append(node)

    output_json = root_path + "/graph.json"
    output_meta = root_path + "/graph_meta.json"

    with open(output_json, "w") as wf:
        for node in nodes:
            ss = json.dumps(node)
            wf.write(ss)
            wf.write("\n")
        wf.close()

    with open(output_meta, "w") as wf_meta:
        meta_data = {
            "node_type_num": 1,
            "edge_type_num": 2,
            "node_uint64_feature_num": 0,
            "node_float_feature_num": 0,
            "node_binary_feature_num": 0,
            "edge_uint64_feature_num": 1,
            "edge_float_feature_num": 0,
            "edge_binary_feature_num": 0,
        }
        str_meta = json.dumps(meta_data)
        wf_meta.write(str_meta)
        wf_meta.close()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Please provide a data path.")
        exit(1)
    main(sys.argv[1])
