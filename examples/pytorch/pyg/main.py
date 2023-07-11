"""PyGeo interface training example.

https://github.com/pyg-team/pytorch_geometric/blob/ae84a38f14591ba9b8ce64e704e04ea1271c3b78/examples/graph_sage_unsup_ppi.py#L12
Epoch: 05, Loss: 0.5647
"""
import numpy as np
import torch
import torch.nn.functional as F
import tqdm  # type: ignore

from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.nn import GraphSAGE
from torch_geometric.data.feature_store import FeatureStore, TensorAttr
from torch_geometric.data.graph_store import GraphStore, EdgeAttr

from deepgnn.graph_engine import SamplingStrategy
from deepgnn.graph_engine.data.citation import Cora
from sklearn.metrics import f1_score


class DeepGNNFeatureStore(FeatureStore):
    """An abstract base class to access features from a remote feature store."""

    def __init__(self, ge):
        """Initialize DeepGNN feature store."""
        super().__init__()
        self.ge = ge

    def _put_tensor(self, tensor, attr) -> bool:
        """To be implemented by :class:`FeatureStore` subclasses."""
        return False

    def _remove_tensor(self, attr) -> bool:
        """To be implemented by :obj:`FeatureStore` subclasses."""
        return False

    def _get_tensor(self, attr):
        """To be implemented by :class:`FeatureStore` subclasses."""
        feature = np.array([[0, 121]]) if attr.attr_name == "x" else np.array([[1, 1]])
        return torch.Tensor(
            self.ge.node_features(attr.index.detach().numpy(), feature, np.float32)
        ).squeeze()

    def _get_tensor_size(self, attr):
        return attr.size()

    def get_all_tensor_attrs(self):
        """Obtain all tensor attributes stored in this :class:`FeatureStore`."""
        output = []
        for i in ["x", "y"]:
            ta = TensorAttr()
            ta.group_name = "0"
            ta.attr_name = f"{i}"
            output.append(ta)
        return output


class DeepGNNGraphStore(GraphStore):
    """An abstract base class to access edges from a remote graph store.

    Args:
        edge_attr_cls (EdgeAttr, optional): A user-defined
            :class:`EdgeAttr` class to customize the required attributes and
            their ordering to uniquely identify edges. (default: :obj:`None`)
    """

    def __init__(self, ge, node_path):
        """Initialize DeepGNN graph store."""
        super().__init__()
        self.ge = ge
        self.node_ids = np.loadtxt(node_path, dtype=np.int64)

    def _put_edge_index(self, edge_index, edge_attr) -> bool:
        """To be implemented by :class:`GraphStore` subclasses."""
        return False

    def _remove_edge_index(self, edge_attr) -> bool:
        """To be implemented by :class:`GraphStore` subclasses."""
        return False

    def _get_edge_index(self, edge_attr):
        """To be implemented by :class:`GraphStore` subclasses."""
        edge_type = int(edge_attr.edge_type[1])
        edge = self.ge.neighbors(self.node_ids, edge_type)
        srcs = []
        for i, e in enumerate(edge[-1]):
            srcs.extend([self.node_ids[i]] * e)
        edge_index = (
            torch.Tensor(srcs).long(),
            torch.Tensor(edge[0]).long(),
        )
        return edge_index

    def get_all_edge_attrs(self):
        """Obtain all edge attributes stored in the :class:`GraphStore`."""
        output = []
        ta = EdgeAttr(("0", "0", "0"), "coo", size=[self.ge.node_count(0), 2708])
        output.append(ta)

        return output


def train():
    """Train the model."""
    model.train()

    total_loss = total_examples = 0
    for data in tqdm.tqdm(loader):
        data = data.to(device)
        data = data[("0", "0", "0")]
        optimizer.zero_grad()

        h = model(data.x, data.edge_index)

        h_src = h[data.edge_label_index[0]]
        h_dst = h[data.edge_label_index[1]]
        link_pred = (h_src * h_dst).sum(dim=-1)  # Inner product.

        loss = F.binary_cross_entropy_with_logits(link_pred, data.edge_label)
        loss.backward()
        optimizer.step()

        total_loss += float(loss) * link_pred.numel()
        total_examples += link_pred.numel()

    return total_loss / total_examples


def test():
    for data in tqdm.tqdm(loader):
        data = data[("0", "0", "0")]
        pred = model(data.x, data.edge_index)
        pred = pred.argmax(1).detach().numpy()
        test_f1 = f1_score(data.y, pred, average="micro")
        return test_f1


if __name__ == "__main__":
    ge = Cora()
    loader = LinkNeighborLoader(
        (
            DeepGNNFeatureStore(ge),
            DeepGNNGraphStore(ge, f"{ge.data_dir()}/train.nodes"),
        ),
        batch_size=140,
        shuffle=True,
        neg_sampling_ratio=1.0,
        num_neighbors=[5, 5],
        num_workers=1,
        persistent_workers=True,
        edge_label_index=(
            ("0", "0", "0"),
            torch.Tensor(ge.sample_edges(2708, np.array(0), SamplingStrategy.Random))
            .long()[:, :2]
            .T,
        ),
    )

    test_loader = LinkNeighborLoader(
        (DeepGNNFeatureStore(ge), DeepGNNGraphStore(ge, f"{ge.data_dir()}/test.nodes")),
        batch_size=140,
        shuffle=True,
        neg_sampling_ratio=1.0,
        num_neighbors=[5, 5],
        num_workers=1,
        persistent_workers=True,
        edge_label_index=(
            ("0", "0", "0"),
            torch.Tensor(ge.sample_edges(2708, np.array(0), SamplingStrategy.Random))
            .long()[:, :2]
            .T,
        ),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GraphSAGE(
        in_channels=121,
        hidden_channels=64,
        num_layers=2,
        out_channels=7,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

    for epoch in range(1, 10):
        loss = train()
        test_f1 = test()
        print(f"Epoch: {epoch:03d}, Loss: {loss:.4f}, Test F1: {test_f1:.4f}")
    assert loss <= 0.55
