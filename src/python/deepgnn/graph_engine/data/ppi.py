# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
import json
import os
import zipfile
from typing import List, Dict

import numpy as np
from sklearn.preprocessing import StandardScaler
import deepgnn.graph_engine.snark.convert as convert
import deepgnn.graph_engine.snark.decoders as decoders

from deepgnn.graph_engine.data.data_util import download_file
from deepgnn.graph_engine.snark.local import Client


class PPI(Client):
    """
    Protein-Protein Interactions graph.
    Use positional gene sets, motif gene sets and immunological signatures as features
    and gene ontology sets as labels (121 in total).

    References:
        http://snap.stanford.edu/graphsage/#datasets
        https://github.com/williamleif/GraphSAGE/blob/master/graphsage/utils.py

    Graph Statistics:
    - Nodes: 56944
    - Edges: 1637432
    - Split: (Train: 44906, Valid: 6514, Test: 5524)
    - Node Label Dim: 121 (id:0)
    - Node Feature Dim: 50 (id:1)
    """

    def __init__(self, output_dir: str = None):
        """
        Args:
          output_dir (string): file directory for graph data.
        """
        self.url = "https://deepgraphpub.blob.core.windows.net/public/testdata/ppi.zip"
        self.GRAPH_NAME = "ppi"
        self.output_dir = output_dir
        if self.output_dir is None:
            self.output_dir = os.path.join("/tmp/", self.GRAPH_NAME)
        self._build_graph(self.output_dir)
        super().__init__(path=self.output_dir, partitions=[0])

    def data_dir(self):
        return self.output_dir

    def _load_raw_graph(self, data_dir: str):
        id_map = json.load(open(os.path.join(data_dir, "ppi-id_map.json")))
        id_map = {int(k): v for k, v in id_map.items()}

        g = json.load(open(os.path.join(data_dir, "ppi-G.json")))
        nodes = []
        nodes_type = []
        train_nodes = []
        NODE_TYPE_ID = {"train": 0, "val": 1, "test": 2}
        for node in g["nodes"]:
            nid = id_map[node["id"]]
            if node["test"]:
                ntype = NODE_TYPE_ID["test"]
            elif node["val"]:
                ntype = NODE_TYPE_ID["val"]
            else:
                ntype = NODE_TYPE_ID["train"]
                train_nodes.append(nid)
            nodes.append(nid)
            nodes_type.append(ntype)

        train_neighbors: Dict = {nid: [] for nid in nodes}
        other_neighbors: Dict = {nid: [] for nid in nodes}

        ## edges
        train_mask = np.zeros(len(nodes), np.bool)
        train_mask[train_nodes] = True
        for i, e in enumerate(g["links"]):
            src = id_map[e["source"]]
            tgt = id_map[e["target"]]
            if train_mask[src] and train_mask[tgt]:
                train_neighbors[src].append(tgt)
                if tgt != src:
                    train_neighbors[tgt].append(src)
            else:
                other_neighbors[src].append(tgt)
                if tgt != src:
                    other_neighbors[tgt].append(src)

        ## class map
        fname = os.path.join(data_dir, "ppi-class_map.json")
        class_map = json.load(open(fname))
        class_map = {id_map[int(k)]: v for k, v in class_map.items()}

        ## feat
        feats = np.load(os.path.join(data_dir, "ppi-feats.npy"))
        train_feats = feats[train_nodes]
        scaler = StandardScaler()
        scaler.fit(train_feats)
        feats = scaler.transform(feats)

        return nodes, nodes_type, train_neighbors, other_neighbors, feats, class_map

    def _build_graph(self, output_dir: str) -> str:
        data_dir = output_dir
        raw_data_dir = os.path.join(output_dir, "raw")
        download_file(self.url, raw_data_dir, "ppi.zip")
        fname = os.path.join(raw_data_dir, "ppi.zip")
        with zipfile.ZipFile(fname) as z:
            z.extractall(raw_data_dir)
        d = self._load_raw_graph(os.path.join(raw_data_dir, "ppi"))
        nodes, nodes_type, train_neighbors, other_neighbors, feats, class_map = d
        assert feats.shape[0] == len(nodes)
        self.NUM_NODES = len(nodes)
        self.FEATURE_DIM = feats.shape[1]
        self.NUM_CLASSES = len(class_map[0])

        graph_file = os.path.join(data_dir, "graph.json")
        with open(graph_file, "w") as fout:
            for i in range(len(nodes)):
                nid = nodes[i]
                ntype = nodes_type[i]
                nfeat = feats[nid].reshape(-1).tolist()
                label = class_map[nid]
                assert type(nfeat) == list and type(nfeat[0]) == float
                assert type(label) == list
                tmp = self._to_json_node(
                    nid,
                    ntype,
                    nfeat,
                    label,
                    train_neighbor=train_neighbors[nid],
                    train_removed_neighbor=other_neighbors[nid],
                )
                fout.write(tmp)
                fout.write("\n")

        meta_file = os.path.join(data_dir, "meta.json")
        self._write_meta_file(meta_file)
        self._write_node_files(data_dir, nodes, nodes_type)
        ## convert graph: JSON -> Binary
        convert.MultiWorkersConverter(
            graph_path=graph_file,
            meta_path=meta_file,
            partition_count=1,
            output_dir=data_dir,
            decoder_type=decoders.DecoderType.JSON,
        ).convert()

        return data_dir

    def _write_node_files(self, data_dir, nodes: List[int], nodes_type: List[int]):
        train_file = os.path.join(data_dir, "train.nodes")
        val_file = os.path.join(data_dir, "val.nodes")
        test_file = os.path.join(data_dir, "test.nodes")
        with open(train_file, "w") as fout_train, open(val_file, "w") as fout_val, open(
            test_file, "w"
        ) as fout_test:
            for nid, ntype in zip(nodes, nodes_type):
                if ntype == 0:
                    fout_train.write(str(nid) + "\n")
                elif ntype == 1:
                    fout_val.write(str(nid) + "\n")
                elif ntype == 2:
                    fout_test.write(str(nid) + "\n")

    def _write_meta_file(self, meta_file: str):
        meta = '{"node_type_num": 3, \
                 "node_float_feature_num": 2, \
                 "node_binary_feature_num": 0, \
                 "node_uint64_feature_num": 0, \
                 "edge_type_num": 2, \
                 "edge_float_feature_num": 0, \
                 "edge_binary_feature_num": 0, \
                 "edge_uint64_feature_num": 0}'

        with open(meta_file, "w") as fout:
            fout.write(meta)

    def _to_json_node(
        self,
        node_id: int,
        node_type: int,
        feat: List[float],
        label: List[float],
        train_neighbor: List[int],
        train_removed_neighbor: List[int],
    ):
        nb0 = {str(nb): 1.0 for nb in train_neighbor}
        nb1 = {str(nb): 1.0 for nb in train_removed_neighbor}
        edges = []
        for nb in train_neighbor:
            e = {"src_id": node_id, "dst_id": nb, "edge_type": 0, "weight": 1.0}
            edges.append(e)

        for nb in train_removed_neighbor:
            e = {"src_id": node_id, "dst_id": nb, "edge_type": 1, "weight": 1.0}
            edges.append(e)

        node = {
            "node_id": node_id,
            "node_type": node_type,
            "node_weight": 1.0,
            "float_feature": {"0": label, "1": feat},
            "edge": edges,
            "neighbor": {"0": nb0, "1": nb1},
        }

        return json.dumps(node)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="/tmp/ppi", type=str)
    args = parser.parse_args()

    g = PPI(args.data_dir)
