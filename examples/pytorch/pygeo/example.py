"""https://github.com/pyg-team/pytorch_geometric/blob/master/examples/compile/gin.py."""
from torch_geometric.loader import NeighborLoader
from torch_geometric.data.feature_store import FeatureStore
from torch_geometric.data.graph_store import GraphStore


class DeepGNNFeatureStore(FeatureStore):
    """An abstract base class to access features from a remote feature store."""

    def __init__(self):
        """Initialize DeepGNN feature store."""
        super().__init__()

    def _put_tensor(self, tensor, attr) -> bool:
        """To be implemented by :class:`FeatureStore` subclasses."""
        return False

    def _remove_tensor(self, attr) -> bool:
        """To be implemented by :obj:`FeatureStore` subclasses."""
        return False

    def _get_tensor(self, attr):
        """To be implemented by :class:`FeatureStore` subclasses."""
        assert False, attr
        return

    def _get_tensor_size(self, attr):
        return attr.size()


class DeepGNNGraphStore(GraphStore):
    """An abstract base class to access edges from a remote graph store.

    Args:
        edge_attr_cls (EdgeAttr, optional): A user-defined
            :class:`EdgeAttr` class to customize the required attributes and
            their ordering to uniquely identify edges. (default: :obj:`None`)
    """

    def __init__(self):
        """Initialize DeepGNN graph store."""
        super().__init__()

    def _put_edge_index(self, edge_index, edge_attr) -> bool:
        """To be implemented by :class:`GraphStore` subclasses."""
        return False

    def _remove_edge_index(self, edge_attr) -> bool:
        """To be implemented by :class:`GraphStore` subclasses."""
        return False

    def _get_edge_index(self, edge_attr):
        """To be implemented by :class:`GraphStore` subclasses."""
        assert False, edge_attr


loader = NeighborLoader(
    (DeepGNNFeatureStore(), DeepGNNGraphStore()),
    # Sample 30 neighbors for each node for 2 iterations
    num_neighbors=[30] * 2,
    # Use a batch size of 128 for sampling training nodes
    batch_size=128,
)

sampled_data = next(iter(loader))
print(sampled_data.batch_size)
