# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Citation graph datasets."""
import argparse
import sys
import tempfile
import os
from typing import Optional, List, Tuple, Union

import deepgnn.graph_engine.snark.convert as convert
import deepgnn.graph_engine.snark.decoders as decoders
import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp  # type: ignore

from deepgnn.graph_engine.data.data_util import download_file
from deepgnn.graph_engine.snark.local import Client


def parse_index_file(filename: str) -> List[int]:
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx: Union[List[int], np.ndarray], size: int) -> np.ndarray:
    """Create mask."""
    mask = np.zeros(size)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool8)


def load_data(
    data_dir: str, dataset_str: str
) -> Tuple[
    sp.csr_matrix,
    sp.lil_matrix,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Load input data from gcn/data directory.

    Reference: https://github.com/tkipf/gcn/blob/master/gcn/utils.py.

    ind.dataset_str.x => the feature vectors of the training instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.tx => the feature vectors of the test instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.allx => the feature vectors of both labeled and unlabeled training instances
        (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.y => the one-hot labels of the labeled training instances as numpy.ndarray object;
    ind.dataset_str.ty => the one-hot labels of the test instances as numpy.ndarray object;
    ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
    ind.dataset_str.graph => a dict in the format {index: [index_of_neighbor_nodes]} as collections.defaultdict
        object;
    ind.dataset_str.test.index => the indices of test instances in graph, for the inductive setting as list object.

    All objects above must be saved using python pickle module.

    :param dataset_str: Dataset name
    :return: All data input files loaded (as well the training/test data).
    """
    names = ["x", "y", "tx", "ty", "allx", "ally", "graph"]
    objects = []
    for i in range(len(names)):
        fname = os.path.join(data_dir, "ind.{}.{}".format(dataset_str, names[i]))
        with open(fname, "rb") as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding="latin1"))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    fname = os.path.join(data_dir, "ind.{}.test.index".format(dataset_str))
    test_idx_reorder = parse_index_file(fname)
    test_idx_range: np.ndarray = np.sort(test_idx_reorder)

    if dataset_str == "citeseer":
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range - min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range.tolist()
    idx_train = list(range(len(y)))
    idx_val = list(range(len(y), len(y) + 500))

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    return (
        adj,
        features,
        labels,
        y_train,
        y_val,
        y_test,
        train_mask,
        val_mask,
        test_mask,
    )


def preprocess_features(features: sp.lil_matrix) -> np.ndarray:
    """Row-normalize feature matrix and convert to tuple representation."""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.0
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features.toarray()


def download_gcn_data(dataset: str, data_dir: str):
    """Download GCN data."""
    url = "https://deepgraphpub.blob.core.windows.net/public/testdata/gcndata"
    names = ["x", "tx", "allx", "y", "ty", "ally", "graph", "test.index"]
    all_files = ["ind.{}.{}".format(dataset.lower(), name) for name in names]
    for name in all_files:
        furl = "{}/{}".format(url, name)
        download_file(furl, data_dir, name)


def random_split(
    labels: np.ndarray,
    num_classes: int,
    num_train_per_class: int,
    num_val: int,
    num_test: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split graph data to train/test/validation sets with node sampling."""
    num_train_per_class = 20
    num_nodes = labels.shape[0]
    train_idx = []
    for cl in range(num_classes):
        idx = (labels == cl).nonzero()[0]
        idx = np.random.permutation(idx)[:num_train_per_class]
        train_idx.append(idx)
    train_idx = np.concatenate(train_idx)
    train_mask = sample_mask(train_idx, num_nodes)

    remaining = (~train_mask).nonzero()[0]
    remaining = np.random.permutation(remaining)
    val_idx = remaining[0:num_val]
    test_idx = remaining[num_val : num_val + num_test]
    val_mask = sample_mask(val_idx, num_nodes)
    test_mask = sample_mask(test_idx, num_nodes)
    return train_mask, val_mask, test_mask


class CitationGraph(Client):
    """Citation graph dataset."""

    def __init__(
        self, name: str, output_dir: Optional[str] = None, split: str = "public"
    ):
        """Initialize dataset."""
        assert name in ["cora", "citeseer"]
        self.GRAPH_NAME = name
        self.output_dir = output_dir
        if self.output_dir is None:
            self.output_dir = os.path.join(
                tempfile.gettempdir(), "citation", self.GRAPH_NAME
            )
        self._build_graph(self.output_dir, split)
        super().__init__(path=self.output_dir, partitions=[0])

    def data_dir(self):
        """Graph location on disk."""
        return self.output_dir

    def _load_raw_graph(self, data_dir: str):
        (adj, features, labels, _, _, _, train_mask, val_mask, test_mask) = load_data(
            data_dir, dataset_str=self.GRAPH_NAME
        )
        features = preprocess_features(features)
        labels = np.argmax(labels, 1)
        self.NUM_CLASSES = max(labels) + 1
        self.NUM_NODES = features.shape[0]
        self.FEATURE_DIM = features.shape[1]
        return adj, features, labels, train_mask, val_mask, test_mask

    def _build_graph(self, output_dir: str, split: str) -> str:
        assert split in ["public", "random"]
        data_dir = output_dir
        raw_data_dir = os.path.join(output_dir, "raw")
        download_gcn_data(self.GRAPH_NAME, raw_data_dir)
        adj, features, labels, train_mask, val_mask, test_mask = self._load_raw_graph(
            raw_data_dir
        )

        if split == "random":
            num_val, num_test = sum(val_mask), sum(test_mask)
            train_mask, val_mask, test_mask = random_split(
                labels, self.NUM_CLASSES, 20, num_val, num_test
            )

        node_types = self.get_node_types(train_mask, val_mask, test_mask)

        graph_file = os.path.join(data_dir, "graph.csv")
        self._write_edge_list_graph(adj, features, labels, node_types, graph_file)

        convert.MultiWorkersConverter(
            graph_path=graph_file,
            partition_count=1,
            output_dir=data_dir,
            decoder=decoders.EdgeListDecoder(),
        ).convert()

        train_file = os.path.join(data_dir, "train.nodes")
        test_file = os.path.join(data_dir, "test.nodes")
        self._write_node_files(node_types, train_file, test_file)
        return output_dir

    def get_node_types(
        self, train_mask: np.ndarray, val_mask: np.ndarray, test_mask: np.ndarray
    ) -> List[str]:
        """Return node types: train/val/test."""
        node_types = []
        for nid in range(train_mask.shape[0]):
            t = "other"
            if train_mask[nid]:
                t = "train"
            elif val_mask[nid]:
                t = "val"
            elif test_mask[nid]:
                t = "test"
            node_types.append(t)
        return node_types

    def _write_edge_list_graph(
        self,
        adj: sp.csr_matrix,
        features: np.ndarray,
        labels: np.ndarray,
        node_types: List[str],
        graph_file: str,
    ):
        node_size = adj.shape[0]
        assert node_size == features.shape[0]
        assert node_size == labels.shape[0]
        with open(graph_file, "w") as fout:
            for nid in range(node_size):
                tmp = self.to_edge_list_node(
                    nid,
                    node_types[nid],
                    flt_feat=list(features[nid].tolist()),
                    label=int(labels[nid]),
                    neighbors=adj[nid].nonzero()[1],
                )
                fout.write(tmp)

    NODE_TYPE_ID = {"train": 0, "val": 1, "test": 2, "other": 3}

    def to_edge_list_node(
        self,
        node_id: int,
        node_type: str,
        flt_feat: List[float],
        label: int,
        neighbors: np.ndarray,
    ) -> str:
        """Convert node to edge_list format."""
        assert type(flt_feat) is list and type(flt_feat[0]) == float
        assert type(label) is int
        ntype = self.NODE_TYPE_ID[node_type]
        output = ""
        output += f"{node_id},-1,{ntype},1.0,float32,{len(flt_feat)},{','.join([str(v) for v in flt_feat])},float32,1,{label}\n"
        for nb in sorted(neighbors):
            output += f"{node_id},0,{nb},1.0\n"
        return output

    def _write_node_files(self, node_types: List[str], train_file: str, test_file: str):
        with open(train_file, "w") as fout_train, open(
            test_file, "w"
        ) as fout_test, open(f"{test_file}.hetgnn", "w") as fout_test_hetgnn:
            for nid, ntype in enumerate(node_types):
                if ntype == "train":
                    fout_train.write(str(nid) + "\n")
                elif ntype == "test":
                    fout_test.write(str(nid) + "\n")
                    fout_test_hetgnn.write(f"{nid},{self.NODE_TYPE_ID[ntype]}\n")


class Cora(CitationGraph):
    """
    The citation network datasets "Cora".

    Args:
      output_dir (string): file directory for graph data.
      split (string): train/val/test data split. ["public", "random"]
          - if set to "public", dataset splits provided by [this paper](https://arxiv.org/abs/1603.08861).
          - if set to "random", train/val/test dataset are randomly generated. each class has 20 training nodes.

    References:
        [GCN data](https://github.com/tkipf/gcn/tree/master/gcn/data).

    Graph Statistics:
    - Nodes: 2708
    - Edges: 10556
    - Number of Classes: 7
    - Split: (Train: 140, Valid: 500, Test: 1000)
    - Node Feature Dim: 1433
    """

    def __init__(self, output_dir: Optional[str] = None, split: str = "public"):
        """Initialize dataset."""
        super().__init__("cora", output_dir, split)


class Citeseer(CitationGraph):
    """
    The citation network datasets "Citeseer".

    Args:
      output_dir (string): file directory for graph data.
      split (string): train/val/test data split. ["public", "random"]
          - if set to "public", dataset splits provided by [this paper](https://arxiv.org/abs/1603.08861).
          - if set to "random", train/val/test dataset are randomly generated. each class has 20 training nodes.

    References:
        [GCN data](https://github.com/tkipf/gcn/tree/master/gcn/data).

    Graph Statistics:
    - Nodes: 3312
    - Edges: 9228
    - Number of Classes: 6
    - Split: (Train: 120, Valid: 500, Test: 1000)
    - Node Feature Dim: 3703
    """

    def __init__(self, output_dir: Optional[str] = None, split: str = "public"):
        """Initialize dataset."""
        super().__init__("citeseer", output_dir, split)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", default="cora", type=str, choices=["cora", "citeseer"]
    )
    parser.add_argument(
        "--data_dir", default=f"{tempfile.gettempdir()}/citation", type=str
    )
    parser.add_argument(
        "--split", default="public", type=str, choices=["public", "random"]
    )
    args = parser.parse_args()

    ds = CitationGraph(args.dataset, args.data_dir, args.split)
    print(f"graph data: {args.data_dir}")
