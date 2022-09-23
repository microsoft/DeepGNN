# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""GAT model implementation."""
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Tuple, Any, Iterator
from torch.utils.data import Dataset, Sampler

from deepgnn.graph_engine.snark.local import Client
from deepgnn.pytorch.common import Accuracy
from deepgnn.pytorch.modeling.base_model import BaseModel
from deepgnn.pytorch.nn.gat_conv import GATConv

from deepgnn.graph_engine import Graph, graph_ops


class GATDataset(Dataset):
    """Cora dataset with file sampler."""
    def __init__(self, data_dir: str, node_types: List[int], feature_meta: List[int], label_meta: List[int], feature_type: np.dtype, label_type: np.dtype, neighbor_edge_types: List[int] = [0], num_hops: int = 2):
        self.g = Client(data_dir, [0, 1])
        self.node_types = np.array(node_types)
        self.feature_meta = np.array([feature_meta])
        self.label_meta = np.array([label_meta])
        self.feature_type = feature_type
        self.label_type = label_type
        self.neighbor_edge_types = np.array(neighbor_edge_types, np.int64)
        self.num_hops = num_hops
        self.count = self.g.node_count(self.node_types)

    def __len__(self):
        return self.count

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        """Query used to generate data for training."""
        inputs = np.array(idx, np.int64)
        nodes, edges, src_idx = graph_ops.sub_graph(
            self.g,
            inputs,
            edge_types=self.neighbor_edge_types,
            num_hops=self.num_hops,
            self_loop=True,
            undirected=True,
            return_edges=True,
        )
        input_mask = np.zeros(nodes.size, np.bool)
        input_mask[src_idx] = True

        feat = self.g.node_features(nodes, self.feature_meta, self.feature_type)
        label = self.g.node_features(nodes, self.label_meta, self.label_type)
        label = label.astype(np.int32)
        edges_value = np.ones(edges.shape[0], np.float32)
        edges = np.transpose(edges)
        adj_shape = np.array([nodes.size, nodes.size], np.int64)

        return nodes, feat, input_mask, label, edges, edges_value, adj_shape


class BatchedSampler:
    def __init__(self, sampler, batch_size):
        self.sampler = sampler
        self.batch_size = batch_size

    def __len__(self):
        return len(self.sampler) // self.batch_size

    def __iter__(self) -> Iterator[int]:
        generator = iter(self.sampler)
        x = []
        while True:
            try:
                for _ in range(self.batch_size):
                    x.append(next(generator))
                yield np.array(x, dtype=np.int64)
                x = []
            except Exception:
                break
        if len(x):
            yield np.array(x, dtype=np.int64)


class FileNodeSampler(Sampler[int]):
    def __init__(self, filename: str):
        self.filename = filename

    def __len__(self) -> int:
        raise NotImplementedError("")

    def __iter__(self) -> Iterator[int]:
        with open(self.filename, "r") as file:
            while True:
                yield int(file.readline())


class GAT(BaseModel):
    """GAT model implementation."""

    def __init__(
        self,
        in_dim: int,
        head_num: List = [8, 1],
        hidden_dim: int = 8,
        num_classes: int = -1,
        ffd_drop: float = 0.0,
        attn_drop: float = 0.0,
    ):
        """Initialize GAT model."""
        super().__init__(np.float32, 0, 0, None)
        self.num_classes = num_classes

        self.out_dim = num_classes

        self.input_layer = GATConv(
            in_dim=in_dim,
            attn_heads=head_num[0],
            out_dim=hidden_dim,
            act=F.elu,
            in_drop=ffd_drop,
            coef_drop=attn_drop,
            attn_aggregate="concat",
        )
        layer0_output_dim = head_num[0] * hidden_dim
        # TODO: support hidden layer
        assert len(head_num) == 2
        self.out_layer = GATConv(
            in_dim=layer0_output_dim,
            attn_heads=head_num[1],
            out_dim=self.out_dim,
            act=None,
            in_drop=ffd_drop,
            coef_drop=attn_drop,
            attn_aggregate="average",
        )

        self.metric = Accuracy()

    def forward(self, inputs):
        """Evaluate model, calculate loss, predictions and extract labels."""
        # fmt: off
        nodes, feat, mask, labels, edges, edges_value, adj_shape = inputs
        nodes = torch.squeeze(nodes)                # [N], N: num of nodes in subgraph
        feat = torch.squeeze(feat)                  # [N, F]
        mask = torch.squeeze(mask)                  # [N]
        labels = torch.squeeze(labels)              # [N]
        edges = torch.squeeze(edges)                # [X, 2], X: num of edges in subgraph
        edges_value = torch.squeeze(edges_value)    # [X]
        adj_shape = torch.squeeze(adj_shape)        # [2]
        # fmt: on

        sp_adj = torch.sparse_coo_tensor(edges, edges_value, adj_shape.tolist())
        h_1 = self.input_layer(feat, sp_adj)
        scores = self.out_layer(h_1, sp_adj)

        labels = labels.type(torch.int64)
        labels = labels[mask]  # [batch_size]
        scores = scores[mask]  # [batch_size]
        pred = scores.argmax(dim=1)
        loss = self.xent(scores, labels)
        return loss, pred, labels
