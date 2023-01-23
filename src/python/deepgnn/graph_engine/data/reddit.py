# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Reddit dataset."""

import argparse
import tempfile
import json
import os
from typing import Optional, Dict

import numpy as np
from sklearn.preprocessing import StandardScaler

from deepgnn.graph_engine.data.ppi import PPI


def _onehot(value, size):
    """Convert value to onehot vector."""
    output = [0] * size
    output[value] = 1
    return output


class Reddit(PPI):
    """
    Reddit Subreddit-Subreddit Interactions graph.

    Sampled 50 large communities and built a post-to-post graph, connecting posts if the same user comments
    on both. For features, we use off-the-shelf 300-dimensional GloVe CommonCrawl word vectors.

    References:
        http://snap.stanford.edu/graphsage/#datasets
        https://github.com/williamleif/GraphSAGE/blob/master/graphsage/utils.py

    Graph Statistics:
    - Nodes: 232,965
    - Edges: 114,618,780 * edge_downsample_pct
    - Split: (Train: 152410, Valid: 23699, Test: 55334)
    - Node Label Dim: 50 (id:0)
    - Node Feature Dim: 300 (id:1)
    """

    def __init__(
        self,
        output_dir: Optional[str] = None,
        edge_downsample_pct: float = 0.1,
        num_partitions: int = 2,
        seed: int = 0,
    ):
        """
        Initialize Reddit dataset.

        Args:
          output_dir (string): file directory for graph data.
          edge_downsample_pct (float, default=.1): Percent of edges to use, default reddit graph has 100M edges.
          num_partitions (int, default=2): Number of partitions
          seed (int, default=0): Seed for random number generation.
        """
        self._edge_downsample_pct = edge_downsample_pct
        self._num_partitions = num_partitions
        np.random.seed(seed)
        self.url = "https://snap.stanford.edu/graphsage/reddit.zip"
        self.GRAPH_NAME = "reddit"
        self.output_dir = output_dir
        if self.output_dir is None:
            self.output_dir = os.path.join(tempfile.gettempdir(), self.GRAPH_NAME)
        self._build_graph(self.output_dir)
        super(PPI, self).__init__(
            path=self.output_dir, partitions=list(range(self._num_partitions))
        )

    def _load_raw_graph(self, data_dir: str):
        id_map = json.load(open(os.path.join(data_dir, "reddit-id_map.json")))
        id_map = {k: v for i, (k, v) in enumerate(id_map.items())}

        g = json.load(open(os.path.join(data_dir, "reddit-G.json")))
        nodes = []
        nodes_type = []
        train_nodes = []
        NODE_TYPE_ID = {"train": 0, "val": 1, "test": 2}
        for nid, node in enumerate(g["nodes"]):
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

        train_neighbors: Dict = {nid: [] for nid in id_map.values()}
        other_neighbors: Dict = {nid: [] for nid in id_map.values()}

        # edges
        edge_downsample_mask = (
            np.random.uniform(size=len(g["links"])) < self._edge_downsample_pct
        )
        train_mask = np.zeros(len(id_map), np.bool8)
        train_mask[train_nodes] = True
        for i, e in enumerate(np.array(g["links"])[edge_downsample_mask]):
            src = e["source"]  # id_map[e["source"]]
            tgt = e["target"]  # id_map[e["target"]]
            if train_mask[src] and train_mask[tgt]:
                train_neighbors[src].append(tgt)
                if tgt != src:
                    train_neighbors[tgt].append(src)
            else:
                other_neighbors[src].append(tgt)
                if tgt != src:
                    other_neighbors[tgt].append(src)

        # class map
        fname = os.path.join(data_dir, "reddit-class_map.json")
        class_map = json.load(open(fname))
        class_map = {id_map[k]: _onehot(v, 50) for k, v in class_map.items()}

        # feat
        feats = np.load(os.path.join(data_dir, "reddit-feats.npy"))
        train_feats = feats[train_nodes]
        scaler = StandardScaler()
        scaler.fit(train_feats)
        feats = scaler.transform(feats)

        return nodes, nodes_type, train_neighbors, other_neighbors, feats, class_map


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir", default=f"{tempfile.gettempdir()}/reddit", type=str
    )
    parser.add_argument("--edge_downsample_pct", default=0.1, type=float)
    parser.add_argument("--num_partitions", default=2, type=int)

    args = parser.parse_args()

    g = Reddit(args.data_dir, args.edge_downsample_pct, args.num_partitions)

    print(g.node_features([1], np.array([[0, 50]]), np.float32))
    print(g.node_features([1], np.array([[1, 300]]), np.float32))
