# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""GAT model implementation with torch geometric."""
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn.functional as F
from typing import List

from deepgnn.pytorch.common import Accuracy
from deepgnn.pytorch.modeling.base_model import BaseModel

from deepgnn.graph_engine import Graph, graph_ops
from torch_geometric.nn import GATConv


class GATGeoDataset(Dataset):
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
            graph,
            inputs,
            edge_types=self.p.neighbor_edge_types,
            num_hops=self.p.num_hops,
            self_loop=True,
            undirected=True,
            return_edges=True,
        )
        input_mask = np.zeros(nodes.size, np.bool)
        input_mask[src_idx] = True

        feat = graph.node_features(nodes, self.feat_meta, self.p.feature_type)
        label = graph.node_features(nodes, self.label_meta, self.p.label_type)
        label = label.astype(np.int32)
        edges = np.transpose(edges)

        graph_tensor = (nodes, feat, edges, input_mask, label)
        return graph_tensor


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
    """GAT model."""

    def __init__(
        self,
        in_dim: int,
        q_param: GATQueryParameter,
        head_num: List = [8, 1],
        hidden_dim: int = 8,
        num_classes: int = -1,
        ffd_drop: float = 0.0,
        attn_drop: float = 0.0,
    ):
        """Initialize model."""
        self.q = GATQuery(q_param)
        super().__init__(np.float32, 0, 0, None)
        self.num_classes = num_classes

        self.out_dim = num_classes

        self.conv1 = GATConv(
            in_channels=in_dim,
            out_channels=hidden_dim,
            heads=head_num[0],
            dropout=0.6,
            concat=True,
        )
        layer0_output_dim = head_num[0] * hidden_dim
        self.conv2 = GATConv(
            in_channels=layer0_output_dim,
            out_channels=self.out_dim,
            heads=1,
            dropout=0.6,
            concat=False,
        )

        self.metric = Accuracy()

    def forward(self, inputs):
        """Calculate loss, make predictions and fetch labels."""
        # fmt: off
        nodes, feat, edge_index, mask, labels = inputs
        nodes = torch.squeeze(nodes)                # [N]
        feat = torch.squeeze(feat)                  # [N, F]
        edge_index = torch.squeeze(edge_index)      # [2, X]
        mask = torch.squeeze(mask)                  # [N]
        labels = torch.squeeze(labels)              # [N]
        # fmt: on

        x = feat
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        scores = self.conv2(x, edge_index)

        labels = labels.type(torch.int64)
        labels = labels[mask]  # [batch_size]
        scores = scores[mask]  # [batch_size]
        pred = scores.argmax(dim=1)
        loss = self.xent(scores, labels)
        return loss, pred, labels
