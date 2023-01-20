# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Graph processing utility functions."""
import os
import logging
import tempfile
import random
from typing import Optional, List, Tuple, Dict, Set, DefaultDict

import urllib.request
import tarfile
import deepgnn.graph_engine.snark.convert as convert
import deepgnn.graph_engine.snark.decoders as decoders

from deepgnn.graph_engine.snark.local import Client


def download_file(url: str, data_dir: str, name: str):
    """Create dir and download data."""
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    fname = os.path.join(data_dir, name)
    if not os.path.exists(fname):
        logging.info(f"download: {fname}")
        urllib.request.urlretrieve(url, fname)


def get_edge_list_node(
    node_id: int,
    node_type: str,
    flt_feat: List[float],
    label: int,
    train_neighbors: Set[int],
    test_neighbors: Set[int],
) -> str:
    """Return node with edge_list format.

    node type: 0(train), 1(test)
    use default value for node_weight(1.0), neighbor weight(1.0)
    """
    assert type(flt_feat) is list and type(flt_feat[0]) == float
    assert type(label) is int
    ntype = 0 if node_type == "train" else 1

    output = ""
    output += f"{node_id},-1,{ntype},1.0,float32,{len(flt_feat)},{','.join([str(v) for v in flt_feat])},float32,1,{label}\n"
    for nb in sorted(train_neighbors):
        output += f"{node_id},0,{nb},1.0\n"
    for nb in sorted(test_neighbors):
        output += f"{node_id},1,{nb},1.0\n"
    return output


def write_node_files(node_types: Dict[int, str], train_file: str, test_file: str):
    """Save node files to disk."""
    with open(train_file, "w") as fout_train, open(test_file, "w") as fout_test:
        for nid, ntype in node_types.items():
            if ntype == "train":
                fout_train.write(str(nid) + "\n")
            else:
                fout_test.write(str(nid) + "\n")


def select_training_test_nodes(
    nodeids: List[int], train_node_ratio: float, random_selection: bool
) -> Dict[int, str]:
    """Generate train/test nodes."""
    random.seed(0)
    max_training_node_id = int(len(nodeids) * train_node_ratio)
    types = ["train"] * max_training_node_id + ["test"] * (
        len(nodeids) - max_training_node_id
    )
    if random_selection:
        random.shuffle(types)

    assert len(nodeids) == len(types)
    node_types = {nid: t for nid, t in zip(sorted(nodeids), types)}
    return node_types


class Dataset(Client):
    """Helper class to generate data for cora and citeseer datasets."""

    def __init__(
        self,
        name: str,
        num_nodes: int,
        feature_dim: int,
        num_classes: int,
        url: str,
        train_node_ratio: float,
        random_selection: bool,
        output_dir: Optional[str] = None,
    ):
        """Initialize Dataset."""
        assert name in ["cora_full", "citeseer_full"]
        self.url = url
        self.GRAPH_NAME = name
        self.NUM_NODES = num_nodes
        self.FEATURE_DIM = feature_dim
        self.NUM_CLASSES = num_classes
        self.output_dir = output_dir
        if self.output_dir is None:
            self.output_dir = os.path.join(
                f"{tempfile.gettempdir()}/citation", self.GRAPH_NAME
            )
        self._build_graph_impl(self.output_dir, train_node_ratio, random_selection)
        super().__init__(self.output_dir, partitions=[0])

    def data_dir(self):
        """Graph location on disk."""
        return self.output_dir

    def _build_graph_impl(
        self, data_dir: str, train_node_ratio: float, random_selection: bool
    ) -> str:
        filename = self.GRAPH_NAME + ".tgz"
        download_file(self.url, data_dir, filename)
        with tarfile.open(os.path.join(data_dir, filename)) as tar:
            tar.extractall(data_dir)
        nodes, node_types, train_adjs, test_adjs = self._load_raw_graph(
            data_dir, train_node_ratio, random_selection
        )

        # build graph - edge_list
        graph_file = os.path.join(data_dir, "graph.csv")
        self._write_edge_list_graph(
            nodes, node_types, train_adjs, test_adjs, graph_file
        )

        # convert graph: edge_list -> Binary
        convert.MultiWorkersConverter(
            graph_path=graph_file,
            partition_count=1,
            output_dir=data_dir,
            decoder=decoders.EdgeListDecoder(),
        ).convert()

        # write training/testing nodes.
        train_file = os.path.join(data_dir, "train.nodes")
        test_file = os.path.join(data_dir, "test.nodes")
        write_node_files(node_types, train_file, test_file)

        self._log_graph_statistic(nodes, node_types)
        return data_dir

    def _load_raw_graph(
        self, data_dir: str, train_node_ratio: float, random_selection: bool
    ):
        raise NotImplementedError()

    def _log_graph_statistic(
        self, nodes: Dict[int, Tuple[List[float], int]], node_types: Dict[int, str]
    ):
        train_nodes = set([nid for nid in nodes if node_types[nid] == "train"])
        test_nodes = set([nid for nid in nodes if node_types[nid] == "test"])
        logging.info("*********** {} graph info *************".format(self.GRAPH_NAME))
        logging.info(
            "* node type: 0, id: {} ~ {}, num: {}".format(
                min(train_nodes), max(train_nodes), len(train_nodes)
            )
        )
        if len(test_nodes) > 0:
            logging.info(
                "* node type: 1, id: {} ~ {}, num: {}".format(
                    min(test_nodes), max(test_nodes), len(test_nodes)
                )
            )
        logging.info("* edge type: 0")
        logging.info("* edge type: 1")
        logging.info("* feature_idx 0")
        logging.info("* feature_dim {}".format(self.FEATURE_DIM))
        logging.info("* label_idx 0")
        logging.info("* label_dim 1")
        logging.info("* classes {}".format(set([node[1] for _, node in nodes.items()])))
        logging.info("*******************************************")

    def _write_edge_list_graph(
        self,
        nodes: Dict[int, Tuple[List[float], int]],
        node_types: Dict[int, str],
        train_adjs: DefaultDict[int, Set[int]],
        test_adjs: DefaultDict[int, Set[int]],
        graph_file: str,
    ):
        with open(graph_file, "w") as fout:
            for nid, info in nodes.items():
                tmp = get_edge_list_node(
                    nid,
                    node_types[nid],
                    info[0],
                    info[1],
                    train_adjs[nid],
                    test_adjs[nid],
                )
                fout.write(tmp)
