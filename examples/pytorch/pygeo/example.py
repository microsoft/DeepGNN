"""https://github.com/pyg-team/pytorch_geometric/blob/master/examples/compile/gin.py."""
import numpy as np
import torch
from torch_geometric.loader import NeighborLoader
from torch_geometric.data.feature_store import FeatureStore, TensorAttr
from torch_geometric.data.graph_store import GraphStore, EdgeAttr
from deepgnn.graph_engine.data.citation import Cora
from torch_geometric.datasets import Planetoid
from deepgnn.graph_engine import SamplingStrategy


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
        edge_index = (torch.Tensor(edge[:, 0]).long(), torch.Tensor(edge[:, 1]).long())
        # assert False, edge_index
        return edge_index

    def get_all_edge_attrs(self):
        """Obtain all edge attributes stored in the :class:`GraphStore`."""
        output = []
        # for i in range(self.ge.node_count(0)):
        # node_type_0, edge_type, node_type_1
        ta = EdgeAttr(
            ("0", "0", "0"), "coo", size=[self.ge.node_count(0), self.ge.edge_count(0)]
        )
        output.append(ta)

        return output


data = Planetoid("../cora", name="Cora")[0]

loader_original = NeighborLoader(
    data,
    # Sample 30 neighbors for each node for 2 iterations
    num_neighbors=[30] * 2,
    # Use a batch size of 128 for sampling training nodes
    batch_size=128,
    input_nodes=data.train_mask,
)

ge = Cora()
loader = NeighborLoader(
    (DeepGNNFeatureStore(ge), DeepGNNGraphStore(ge)),
    num_neighbors=[30] * 2,
    batch_size=128,
    input_nodes=("0", [i for i in range(140)]),
)

sampled_data_original = next(iter(loader_original))
print(sampled_data_original.to_heterogeneous())

sampled_data = next(iter(loader))
print(sampled_data)
# print(sampled_data.to_homogeneous())
