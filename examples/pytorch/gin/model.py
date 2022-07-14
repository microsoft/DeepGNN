from multiprocessing.sharedctypes import Value
from optparse import Option
from typing import Optional

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from deepgnn.graph_engine import Graph, FeatureType
from deepgnn.pytorch.common import MeanAggregator, BaseMetric, MRR
from deepgnn.pytorch.modeling import BaseSupervisedModel, BaseUnsupervisedModel
from deepgnn.pytorch.encoding import FeatureEncoder
from deepgnn import get_logger

class MLP(nn.Module):
    """Simple MLP with linear-output model."""

    def __init__(
        self,
        num_layers: int,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
    ):
        super(MLP, self).__init__()
        self.num_layers = num_layers

        if num_layers < 1:
            raise ValueError("Num layers should be > 0")
        elif num_layers == 1:
            # Single linear model
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            # Multi-layer model
            self.linears = torch.nn.ModuleList()
            self.batch_norms = torch.nn.ModuleList()

            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))

            self.linears.append(nn.Linear(hidden_dim, output_dim))

            for layer in range(num_layers - 1):
                self.batch_norms.append(nn.BatchNorm1d((hidden_dim)))

    def forward(self, x):
        if self.num_layers == 1:
            return self.linear(x)
        else:
            h = x
            for layer in range(self.num_layers - 1):
                h = F.relu(self.batch_norms[layer][self.linears[layer](h)])
            return self.linears[self.num_layers - 1](h)


class GIN(nn.Module):
    """Simple supervised GIN model."""

    '''
            num_layers: number of layers in the neural networks (INCLUDING the input layer)
            num_mlp_layers: number of layers in mlps (EXCLUDING the input layer)
            input_dim: dimensionality of input features
            hidden_dim: dimensionality of hidden units at ALL layers
            output_dim: number of classes for prediction
            final_dropout: dropout ratio on the final linear layer
            learn_eps: If True, learn epsilon to distinguish center nodes from neighboring nodes. If False, aggregate neighbors and center nodes altogether. 
            neighbor_pooling_type: how to aggregate neighbors (mean, average, or max)
            graph_pooling_type: how to aggregate entire nodes in a graph (mean, average)
            device: which device to use
        '''

    def __init__(
        self, 
        num_layers: int, 
        num_mlp_layers: int, 
        edge_type: np.ndarray,
        input_dim:int,
        label_idx: int,
        label_dim: int,
        hidden_dim: int, 
        output_dim: int,
        feature_type: FeatureType,
        feature_idx: int,
        feature_dim: int, 
        final_dropout: str, 
        learn_eps: bool, 
        graph_pooling_type: str,
        neighbor_pooling_type: str, 
        device: None,
    ):

        super(GIN, self).__init__()

        self.final_dropout = final_dropout
        self.device = device
        self.edge_type = edge_type
        self.num_layers = num_layers
        self.graph_pooling_type = graph_pooling_type
        self.neighbor_pooling_type = neighbor_pooling_type
        self.learn_eps = learn_eps
        self.eps = nn.Parameter(torch.zeros(self.num_layers-1))

        self.label_idx = label_idx
        self.label_dim = label_dim
        self.feature_type = feature_type
        self.feature_idx = feature_idx
        self.feature_dim = feature_dim

        ###List of MLPs
        self.mlps = torch.nn.ModuleList()

        ###List of batchnorms applied to the output of MLP (input of the final prediction linear layer)
        self.batch_norms = torch.nn.ModuleList()

        for layer in range(self.num_layers-1):
            if layer == 0:
                self.mlps.append(MLP(num_mlp_layers, input_dim, hidden_dim, hidden_dim))
            else:
                self.mlps.append(MLP(num_mlp_layers, hidden_dim, hidden_dim, hidden_dim))

            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        #Linear function that maps the hidden representation at dofferemt layers into a prediction score
        self.linears_prediction = torch.nn.ModuleList()
        for layer in range(num_layers):
            if layer == 0:
                self.linears_prediction.append(nn.Linear(input_dim, output_dim))
            else:
                self.linears_prediction.append(nn.Linear(hidden_dim, output_dim))

    def graphpool(self, batch_graph):
        """"Build sum/average pooling matrix over all nodes."""
        start_idx = [0]

        #compute the padded neighbor list
        for i, graph in enumerate(batch_graph):
            start_idx.append(start_idx[i] + 1)

        idx = []
        elem = []
        for i, graph in enumerate(batch_graph):
            ###average pooling
            if self.graph_pooling_type == "average":
                elem.extend([1./1]*1)
            else:
            ###sum pooling
                elem.extend([1]*1)

            idx.extend([[i, j] for j in range(start_idx[i], start_idx[i+1], 1)])
        elem = torch.FloatTensor(elem)
        idx = torch.LongTensor(idx).transpose(0,1)
        graph_pool = torch.sparse.FloatTensor(idx, elem, torch.Size([len(batch_graph), start_idx[-1]]))
        
        return graph_pool.to(self.device)
    
    def next_layer(self, h, layer, context):

        # neighbors = context["neighbors"]
        # get_logger().info(f"-"+str(neighbors.shape))


        pooled = sum(context['features'])

        # Sum Pooling
        # feature_matrix = torch.cat([context["features"]], 0).to(self.device)
        # edge_matrix = torch.cat([context["neighbors"]], 0).to(self.device)
        # pooled = torch.spmm(feature_matrix, edge_matrix)

        if self.neighbor_pooling_type == "average":
            degree = sum(context['label'])
            pooled = pooled / degree

        # Average Pooling
        # if self.neighbor_pooling_type == "average":
            #degree = torch.cat([context["label"]], 0).to(self.device)
            #pooled = pooled/degree

        #representation of neighboring and center nodes 
        pooled_rep = self.mlps[layer](pooled)
        h = self.batch_norms[layer](pooled_rep)
        h = F.relu(h)
        return h

    def forward(self, context):
        get_logger().info(f"inside forward.")
        
        X_concat = torch.cat([context["features"]], 0).to(self.device)
        nodes = context["inputs"].squeeze()
        batch_size = len(nodes)

        graph_pool = self.graphpool(nodes)

        # list of hidden representation at each layer (including input)
        hidden_rep = [X_concat]
        h = X_concat
        get_logger().info(f"-"+str(h.shape))

        for layer in range(self.num_layers - 1):
            h = self.next_layer(h, layer, context)
            hidden_rep.append(h)

        score_over_layer = 0
    
        # perform pooling over all nodes in each graph in every layer
        for layer, h in enumerate(hidden_rep):
            pooled_h = torch.spmm(graph_pool, h)
            score_over_layer += F.dropout(self.linears_prediction[layer](pooled_h), self.final_dropout, training = self.training)

        return score_over_layer

    def query(self, graph: Graph, inputs: np.array):
        """Fetch training data from graph."""
        context = {"inputs": inputs}

        get_logger().info(str((type(graph))))
        get_logger().info(str((dir(graph))))

        # context["label"] = graph.neighbor_count(
        #     nodes = context["inputs"],
        #     edge_types = np.ndarray(0),
        # )


        context['neighbors'] = graph.neighbors(
            nodes = context['inputs'],
            edge_types = np.ndarray(0),
        )[0]

        context["features"] = graph.node_features(
            context["neighbors"],
            np.array([[self.label_idx, self.label_dim]]),
            FeatureType.INT64,
        )
     
        return context

    

