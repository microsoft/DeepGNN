# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Example of TGN model trained on MOOC dataset."""
from dataclasses import dataclass
import os
from typing import Tuple

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
import torch
from torch.nn import Linear
from torch_geometric.nn import TGNMemory, TransformerConv
from torch_geometric.nn.models.tgn import (
    IdentityMessage,
    LastAggregator,
    TimeEncoder,
)
from torch.utils.data import DataLoader

from deepgnn.graph_engine import Graph
from deepgnn.graph_engine.data.mooc import MOOC
from deepgnn import get_logger


class GraphAttentionEmbedding(torch.nn.Module):
    """Temporal attention layer."""

    def __init__(
        self, in_channels: int, out_channels: int, msg_dim: int, time_enc: TimeEncoder
    ):
        """Initialize temporal attention layer."""
        super().__init__()
        self.time_enc = time_enc
        edge_dim = msg_dim + time_enc.out_channels
        self.conv = TransformerConv(
            in_channels, out_channels // 2, heads=2, dropout=0.1, edge_dim=edge_dim
        )

    def forward(self, x, last_update, edge_index, t, msg):
        """Compute node embeddings."""
        rel_t = last_update[edge_index[0]] - t
        rel_t_enc = self.time_enc(rel_t.to(x.dtype))
        edge_attr = torch.cat([rel_t_enc, msg], dim=-1)
        return self.conv(x, edge_index, edge_attr)


class LinkPredictor(torch.nn.Module):
    """Compute probabilities for edges between sources and destinations."""

    def __init__(self, in_channels):
        """Initialize link predictor."""
        super().__init__()
        self.lin_src = Linear(in_channels, in_channels)
        self.lin_dst = Linear(in_channels, in_channels)
        self.lin_final = Linear(in_channels, 1)

    def forward(self, z_src, z_dst):
        """Compute edge probabilities."""
        h = self.lin_src(z_src) + self.lin_dst(z_dst)
        h = h.relu()
        return self.lin_final(h)


@dataclass
class Domain:
    """Class to hold model components needed for predictions."""

    assoc: torch.Tensor  # Helper tensors to map global node indices to local ones.
    last_n_assoc: torch.Tensor
    memory: TGNMemory
    gnn: GraphAttentionEmbedding
    link_pred: LinkPredictor


def make_train_val_test_batches(config: dict) -> Tuple[np.array, np.array, np.array]:
    """Split the graph edges into train/val/test batches."""
    edges = np.loadtxt(
        os.path.join(
            config["data_dir"],
            "raw",
            config["graph_name"],
            f'{config["graph_name"]}.csv',
        ),
        delimiter=",",
        skiprows=1,
        usecols=(0, 1, 2),
    ).astype(np.int64)

    edges[:, 1] += config["max_src_id"]
    val_time, test_time = np.quantile(
        edges[:, 2],
        [1 - config["test_ratio"] - config["val_ratio"], 1 - config["test_ratio"]],
    )

    test_idx = (edges[:, 2] <= test_time).sum()
    val_idx = (edges[:, 2] <= val_time).sum()
    return edges[:val_idx], edges[val_idx:test_idx], edges[test_idx:]


def predict(
    batch: torch.Tensor,
    domain: Domain,
    graph: Graph,
    config: dict,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Make predictions for a batch of edges."""
    batch = batch.to(config["device"])
    src, pos_dst, t = batch[:, 0], batch[:, 1], batch[:, 2]
    # MOOC and other typical temporal graphs are bipartite. Destinations node ids
    # are always greater than source node ids. We need to use the right distribution
    # to sample negatives.
    neg_dst = np.random.randint(
        config["min_dst_idx"], config["max_dst_idx"] + 1, (src.size(0),), dtype=np.int64
    )
    n_id, index = np.unique(np.concatenate([src, pos_dst, neg_dst]), return_index=True)
    times = np.concatenate([t, np.ones_like(pos_dst), np.ones_like(neg_dst)])

    # We need to make a prediction based on past interactions of the source node.
    t_id = times[index] - 1
    dst = graph.sample_neighbors(
        n_id,
        0,
        config["num_neighbors"],
        strategy="lastn",
        timestamps=t_id,
        return_edge_created_ts=True,
    )

    nbs = dst[0].ravel()
    # Filter out stubs: if a node has no neighbors, sample_neighbors backfills with -1s.
    dst_ids = nbs[nbs > -1]
    src_np = n_id.repeat(config["num_neighbors"])[nbs > -1]
    dst_ts = dst[3].ravel()[nbs > -1]  # type: ignore

    final_nid = torch.as_tensor(
        np.unique(np.concatenate([n_id, dst_ids])), device=config["device"]
    )
    domain.last_n_assoc[final_nid] = torch.arange(
        final_nid.size(0), device=final_nid.device
    )
    neighbors, nodes = domain.last_n_assoc[dst_ids], domain.last_n_assoc[src_np]
    if nodes.size(0) == 0:
        return (None, None)

    edge_index = torch.stack([neighbors, nodes])
    domain.assoc[final_nid] = torch.arange(final_nid.size(0), device=config["device"])
    edges = np.stack([src_np, dst_ids, np.zeros(src_np.size, dtype=np.int64)], axis=1)
    features_np = graph.edge_features(
        edges=edges,
        features=np.array([[0, config["edge_feature_dim"]]], dtype=np.int32),
        feature_type=np.float32,
        timestamps=dst_ts,
    )

    z, last_update = domain.memory(
        torch.as_tensor(final_nid, dtype=torch.int64).to(config["device"])
    )
    features_torch = torch.as_tensor(features_np).to(config["device"])
    ts_torch = torch.as_tensor(dst_ts).to(config["device"])
    z = domain.gnn(
        z,
        last_update,
        edge_index,
        ts_torch,
        features_torch,
    )

    pos_out = domain.link_pred(z[domain.assoc[src]], z[domain.assoc[pos_dst]])
    neg_out = domain.link_pred(z[domain.assoc[src]], z[domain.assoc[neg_dst]])
    domain.memory.update_state(
        torch.as_tensor(src_np, device=config["device"]),
        torch.as_tensor(dst_ids, device=config["device"]),
        ts_torch,
        features_torch,
    )

    return (pos_out, neg_out)


def train(
    domain: Domain,
    train_loader: DataLoader,
    graph: Graph,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.BCEWithLogitsLoss,
    config: dict,
):
    """Train the domain."""
    domain.memory.train()
    domain.gnn.train()
    domain.link_pred.train()
    domain.memory.reset_state()

    total_loss = 0
    num_examples = len(train_loader) * config["batch_size"]
    for batch in train_loader:
        optimizer.zero_grad()
        pos_out, neg_out = predict(batch, domain, graph, config)
        if pos_out is None:
            continue

        loss = criterion(pos_out, torch.ones_like(pos_out))
        loss += criterion(neg_out, torch.zeros_like(neg_out))

        loss.backward()
        optimizer.step()
        domain.memory.detach()
        total_loss += float(loss) * config["batch_size"]
    return total_loss / num_examples


@torch.no_grad()
def test(
    domain: Domain,
    loader: DataLoader,
    graph: Graph,
    config: dict,
):
    """Evaluate the model on test dataset."""
    domain.memory.eval()
    domain.gnn.eval()
    domain.link_pred.eval()
    aps, aucs = [], []
    for batch in loader:
        batch = batch.to(config["device"])
        pos_out, neg_out = predict(batch, domain, graph, config)
        y_pred = torch.cat([pos_out, neg_out], dim=0).sigmoid().cpu()
        y_true = torch.cat(
            [torch.ones(pos_out.size(0)), torch.zeros(neg_out.size(0))], dim=0
        )

        aps.append(average_precision_score(y_true, y_pred))
        aucs.append(roc_auc_score(y_true, y_pred))

    return float(torch.tensor(aps).mean()), float(torch.tensor(aucs).mean())


def _main():
    graph = MOOC()
    config = {
        "data_dir": graph.data_dir(),
        "graph_name": graph.GRAPH_NAME,
        "max_src_id": graph.max_src,
        "test_ratio": 0.1,
        "val_ratio": 0.1,
        "max_dst_idx": graph.max_dst_idx,
        "min_dst_idx": graph.min_dst_idx,
        "edge_feature_dim": graph.edge_feature_dim,
        "device": torch.device("cpu"),
        "batch_size": 200,
        "num_neighbors": 10,
    }

    train_data, _, test_data = make_train_val_test_batches(config)
    train_loader = DataLoader(train_data, batch_size=config["batch_size"])
    test_loader = DataLoader(test_data, batch_size=config["batch_size"])
    memory_dim = time_dim = embedding_dim = 100
    memory = TGNMemory(
        graph.num_nodes,
        config["edge_feature_dim"],
        memory_dim,
        time_dim,
        message_module=IdentityMessage(
            config["edge_feature_dim"], memory_dim, time_dim
        ),
        aggregator_module=LastAggregator(),
    )
    domain = Domain(
        memory=memory,
        link_pred=LinkPredictor(in_channels=embedding_dim).to(config["device"]),
        assoc=torch.empty(
            graph.num_nodes + 1, dtype=torch.long, device=config["device"]
        ),
        last_n_assoc=torch.empty(
            graph.num_nodes + 1, dtype=torch.long, device=config["device"]
        ),
        gnn=GraphAttentionEmbedding(
            in_channels=memory_dim,
            out_channels=embedding_dim,
            msg_dim=config["edge_feature_dim"],
            time_enc=memory.time_enc,
        ).to(config["device"]),
    )

    optimizer = torch.optim.Adam(
        set(domain.memory.parameters())
        | set(domain.gnn.parameters())
        | set(domain.link_pred.parameters()),
        lr=0.0001,
    )
    criterion = torch.nn.BCEWithLogitsLoss()

    test_ap = 0
    test_auc = 0
    logger = get_logger()
    # We can train the model for 2 epochs to get a decent performance.
    for epoch in range(1, 3):
        loss = train(
            domain,
            train_loader,
            graph,
            optimizer,
            criterion,
            config,
        )
        logger.info(f"Epoch #{epoch:02d}: loss: {loss:.4f}")

    test_ap, test_auc = test(domain, test_loader, graph, config)
    logger.info(f"Average precision: {test_ap:.4f}; AUC: {test_auc:.4f}")
    assert test_ap >= 0.72
    assert test_auc >= 0.76


if __name__ == "__main__":
    torch.manual_seed(42)
    _main()
