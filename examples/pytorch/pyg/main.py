"""PyGeo interface training example."""
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
from deepgnn import get_logger


class DeepGNNFeatureStore(FeatureStore):
    """A class to access features from a DeepGNN graph engine.

    Args:
        ge: MemoryGraph The graph engine to sample from.
    """

    def __init__(self, ge):
        """Initialize DeepGNN feature store."""
        super().__init__()
        self.ge = ge

    def _put_tensor(self, tensor, attr) -> bool:
        """Put tensor."""
        return False

    def _remove_tensor(self, attr) -> bool:
        """Remove tensor."""
        return False

    def _get_tensor(self, attr):
        """Get tensor."""
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
    """A class to access edges from a DeepGNN graph engine.

    Args:
        ge: MemoryGraph The graph engine to sample from.
        node_path: str Path of nodes file to sample initial nodes from.
    """

    def __init__(self, ge, node_path):
        """Initialize DeepGNN graph store."""
        super().__init__()
        self.ge = ge
        self.node_ids = np.loadtxt(node_path, dtype=np.int64)

    def _put_edge_index(self, edge_index, edge_attr) -> bool:
        """Put edge index."""
        return False

    def _remove_edge_index(self, edge_attr) -> bool:
        """Remove edge index."""
        return False

    def _get_edge_index(self, edge_attr):
        """Get edge index."""
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
        ta = EdgeAttr(
            ("0", "0", "0"),
            "coo",
            size=[self.ge.node_count(0), self.ge.node_count(np.array([0, 1, 2, 3]))],
        )
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
    """Test the model."""
    model.eval()
    for data in tqdm.tqdm(loader):
        data = data[("0", "0", "0")]
        h = model(data.x, data.edge_index)
        h_src = h[data.edge_label_index[0]]
        h_dst = h[data.edge_label_index[1]]
        link_pred = (h_src * h_dst).sum(dim=-1)  # Inner product.
        test_f1 = np.mean(
            (link_pred > 0).detach().numpy() == data.edge_label.detach().numpy()
        )
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
            torch.Tensor(
                ge.sample_edges(ge.edge_count(0), np.array(0), SamplingStrategy.Random)
            )
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
            torch.Tensor(
                ge.sample_edges(ge.edge_count(0), np.array(0), SamplingStrategy.Random)
            )
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
        get_logger().info(
            f"Epoch: {epoch:03d}, Loss: {loss:.4f}, Test F1: {test_f1:.4f}"
        )
    assert loss <= 0.55 and test_f1 >= 0.62
