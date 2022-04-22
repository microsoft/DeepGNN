# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
import os
from collections import defaultdict
from deepgnn.graph_engine.data.data_util import Dataset, select_training_test_nodes
from typing import List, Tuple, Dict, Set, DefaultDict


class CiteseerFull(Dataset):
    """
    The citation network datasets "citeseer".
    Args:
      output_dir (string): file directory for graph data.
      train_node_ratio (float): the ratio of training nodes.
      random_selection: train/test data splits
         - if set to True,
         - if set to False, training nodes `[1, X]`, test nodes: `[X+1, NUM_NODES]`.
           here, `X` is NUM_NODES * train_node_ratio.

    Graph Statistics:
    - Nodes: 3312
    - Edges: 9228
    - Number of Classes: 6
    - Node Feature Dim: 3703
    """

    def __init__(
        self,
        output_dir: str = None,
        train_node_ratio: float = 1.0,
        random_selection: bool = False,
        url="https://deepgraphpub.blob.core.windows.net/public/testdata/citeseer.tgz",
    ):
        super().__init__(
            name="citeseer_full",
            num_nodes=3312,
            feature_dim=3703,
            num_classes=6,
            url=url,
            train_node_ratio=train_node_ratio,
            random_selection=random_selection,
            output_dir=output_dir,
        )

    def _load_raw_graph(
        self, data_dir: str, train_node_ratio: float, random_selection: bool
    ) -> Tuple[
        Dict[int, Tuple[List[float], int]],
        Dict[int, str],
        DefaultDict[int, Set[int]],
        DefaultDict[int, Set[int]],
    ]:
        node_file = os.path.join(data_dir, "citeseer", "citeseer.content")
        edge_file = os.path.join(data_dir, "citeseer", "citeseer.cites")
        paper_nodeid = {}
        labelid = {}  # type: ignore
        nodes = {}
        with open(node_file) as fin:
            # node id starts from 1
            nid = 1
            for line in fin:
                col = line.strip().split()
                assert len(col) == 1 + self.FEATURE_DIM + 1
                paper, lb = col[0], col[-1]
                assert paper not in paper_nodeid
                paper_nodeid[paper] = nid
                feat = [float(x) for x in col[1 : 1 + self.FEATURE_DIM]]
                if lb not in labelid:
                    labelid[lb] = len(labelid)
                nodes[nid] = (feat, labelid[lb])
                nid += 1

        nodeids = list(range(1, nid))
        node_types = select_training_test_nodes(
            nodeids, train_node_ratio, random_selection
        )

        train_adj_lists = defaultdict(set)
        test_adj_lists = defaultdict(set)
        with open(edge_file) as fin:
            for i, line in enumerate(fin):
                info = line.strip().split()
                if info[0] not in paper_nodeid:
                    continue
                if info[1] not in paper_nodeid:
                    continue
                paper1 = paper_nodeid[info[0]]
                paper2 = paper_nodeid[info[1]]
                if node_types[paper1] == "train" and node_types[paper2] == "train":
                    train_adj_lists[paper1].add(paper2)
                    train_adj_lists[paper2].add(paper1)
                else:
                    test_adj_lists[paper1].add(paper2)
                    test_adj_lists[paper2].add(paper1)

        return nodes, node_types, train_adj_lists, test_adj_lists


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="/tmp/citeseer", type=str)
    parser.add_argument("--train_node_ratio", default=1.0, type=float)
    parser.add_argument("--random_selection", action="store_true")
    args = parser.parse_args()

    g = CiteseerFull(args.data_dir, args.train_node_ratio, args.random_selection)
    print(f"graph data: {args.data_dir}")
