"""PyGeo interface training example.

https://github.com/pyg-team/pytorch_geometric/blob/ae84a38f14591ba9b8ce64e704e04ea1271c3b78/examples/graph_sage_unsup_ppi.py#L12
Epoch: 05, Loss: 0.5647
"""
import os.path as osp

import numpy as np
import torch
import torch.nn.functional as F
import tqdm  # type: ignore

from torch_geometric.data import Batch
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.nn import GraphSAGE
from torch_geometric.data.feature_store import FeatureStore, TensorAttr
from torch_geometric.data.graph_store import GraphStore, EdgeAttr

from deepgnn.graph_engine import SamplingStrategy

if False:
    from torch_geometric.datasets import PPI

    path = osp.join(osp.dirname(osp.realpath(__file__)), "..", "data", "PPI")
    train_dataset = PPI(path, split="train")
    val_dataset = PPI(path, split="val")
    test_dataset = PPI(path, split="test")

    # Group all training graphs into a single graph to perform sampling:
    train_data = Batch.from_data_list(train_dataset)
    loader = LinkNeighborLoader(
        train_data,
        batch_size=2048,
        shuffle=True,
        neg_sampling_ratio=1.0,
        num_neighbors=[10, 10],
        num_workers=6,
        persistent_workers=True,
    )
else:
    from deepgnn.graph_engine.data.ppi import PPI

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
            feature = (
                np.array([[1, 50]]) if attr.attr_name == "x" else np.array([[0, 121]])
            )
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

        def __init__(self, ge):
            """Initialize DeepGNN graph store."""
            super().__init__()
            self.ge = ge

        def _put_edge_index(self, edge_index, edge_attr) -> bool:
            """To be implemented by :class:`GraphStore` subclasses."""
            return False

        def _remove_edge_index(self, edge_attr) -> bool:
            """To be implemented by :class:`GraphStore` subclasses."""
            return False

        def _get_edge_index(self, edge_attr):
            """To be implemented by :class:`GraphStore` subclasses."""
            edge_type = int(edge_attr.edge_type[1])
            edge = self.ge.sample_edges(
                self.ge.edge_count(0), np.array(edge_type), SamplingStrategy.Random
            )
            edge_index = (
                torch.Tensor(edge[:, 0]).long(),
                torch.Tensor(edge[:, 1]).long(),
            )
            # assert False, edge_index
            return edge_index

        def get_all_edge_attrs(self):
            """Obtain all edge attributes stored in the :class:`GraphStore`."""
            output = []
            # for i in range(self.ge.node_count(0)):
            # node_type_0, edge_type, node_type_1
            ta = EdgeAttr(("0", "0", "0"), "coo", size=[self.ge.node_count(0), 56944])
            output.append(ta)

            return output

    ge = PPI()
    loader = LinkNeighborLoader(
        (DeepGNNFeatureStore(ge), DeepGNNGraphStore(ge)),
        batch_size=2048,
        shuffle=True,
        neg_sampling_ratio=1.0,
        num_neighbors=[10, 10],
        num_workers=6,
        persistent_workers=True,
        edge_label_index=(
            ("0", "0", "0"),
            torch.Tensor(ge.sample_edges(248388, np.array(0), SamplingStrategy.Random))
            .long()[:, :2]
            .T,
        ),
    )


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GraphSAGE(
    in_channels=50,
    hidden_channels=64,
    num_layers=2,
    out_channels=64,
).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)


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


@torch.no_grad()
def encode(loader):
    """Encode the dataloader query."""
    model.eval()

    xs, ys = [], []
    for data in loader:
        data = data.to(device)
        xs.append(model(data.x, data.edge_index).cpu())
        ys.append(data.y.cpu())
    return torch.cat(xs, dim=0), torch.cat(ys, dim=0)


for epoch in range(1, 6):
    loss = train()
    print(f"Epoch: {epoch:02d}, Loss: {loss:.4f}")
