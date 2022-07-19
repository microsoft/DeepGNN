from multiprocessing import pool
from multiprocessing.sharedctypes import Value
from optparse import Option
from typing import Optional

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from deepgnn.graph_engine import Graph, FeatureType
from deepgnn.pytorch.common import MeanAggregator, BaseMetric, MRR
from deepgnn.pytorch.modeling import BaseSupervisedModel
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
        get_logger().info("inside mlp forward call!")
        if self.num_layers == 1:
            get_logger().info("inside solo layer")
            get_logger().info("x: " + str(x))
            return self.linear(x)
        else:
            h = x
            for layer in range(self.num_layers - 1):
                h = F.relu(self.batch_norms[layer][self.linears[layer](h)])
            return self.linears[self.num_layers - 1](h)


class GIN(BaseSupervisedModel):
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
        metric: BaseMetric = MRR(),
        feature_enc: Optional[FeatureEncoder] = None,
    ):

        super(GIN, self).__init__(
            feature_type=feature_type,
            feature_idx=feature_idx,
            feature_dim=feature_dim,
            feature_enc=feature_enc,
        )

        self.final_dropout = final_dropout
        self.device = device
        self.metric = metric
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

    def graphpool(self, batch_graph, context):
        """"Build sum/average pooling matrix over all nodes."""

        start_idx = [0]

        #compute the padded neighbor list
        for i, graph in enumerate(batch_graph):
            # get_logger().info(str(context['features']))
            get_logger().info(str(len(context['features'][0])))
            num_neighbors = 1
            # get_logger().info("Spec: " + str(context['features'][0][graph]))
            # get_logger().info('Num Neighbors: ' + str(num_neighbors))
            start_idx.append(start_idx[i] + num_neighbors)

        idx = []
        elem = []
        for i, graph in enumerate(batch_graph):
            ###average pooling
            if self.graph_pooling_type == "average":
                elem.extend([1./num_neighbors]*num_neighbors)
            
            else:
            ###sum pooling
                # print("len of graph g: " + str(num_neighbors))
                elem.extend([1]*num_neighbors)

            idx.extend([[i, j] for j in range(start_idx[i], start_idx[i+1], 1)])
        elem = torch.FloatTensor(elem)
        idx = torch.LongTensor(idx).transpose(0,1)
        graph_pool = torch.sparse.FloatTensor(idx, elem, torch.Size([len(batch_graph), start_idx[-1]]))
        
        return graph_pool.to(self.device)
    
    def next_layer(self, h, layer, context):
        

        get_logger().info("inside next layer func call.")


        get_logger().info("Neighbors: " + str(context['neighbors']))
        get_logger().info("Features: " + str(context['features']))

        # Sum Pooling
        # for i in range(len())
        pooled = torch.sum(context['counts'])

        get_logger().info("*********************************************")
        get_logger().info("Pooled: " + str(pooled))
        get_logger().info("Pooled Shape: " + str(pooled.shape))
        get_logger().info("*********************************************")


        # Average Pooling
        if self.neighbor_pooling_type == "average":
            degree = sum(context['label'])
            pooled = pooled / degree
        

        #representation of neighboring and center nodes 
        pooled_rep = self.mlps[layer](pooled)
        get_logger().info("*********************************************")
        get_logger().info("Pooled Rep: " + str(pooled))
        get_logger().info("Pooled Rep Shape: " + str(pooled.shape))
        get_logger().info("*********************************************")
        h = self.batch_norms[layer](pooled_rep)
        h = F.relu(h)
        get_logger().info(str(h.shape))
        get_logger().info("finish next layer call")
        return h

    def forward(self, context):
        get_logger().info(f"inside forward.")
        
        X_concat = torch.cat([context["features"]], 0).to(self.device)
        nodes = context["inputs"].squeeze()
        batch_size = len(nodes)

        get_logger().info("Batch Size: " + str(batch_size))

        graph_pool = self.graphpool(nodes, context)

        get_logger().info("finish building graphpool.")

        # list of hidden representation at each layer (including input)
        hidden_rep = [X_concat]
        h = X_concat

        get_logger().info("h: " + str(h))
        get_logger().info("h shape: " + str(h.shape))

        for layer in range(self.num_layers - 1):
            get_logger().info(f"inside layer loop")
            h = self.next_layer(h, layer, context)
            hidden_rep.append(h)

        score_over_layer = 0

       # get_logger().info("hidden rep shape- " + str(hidden_rep.shape))

        # perform pooling over all nodes in each graph in every layer
        for layer, h in enumerate(hidden_rep):
            # get_logger().info("Graph Pool Shape: " + str(graph_pool.shape))
            # get_logger().info("h Shape: " + str(h.shape))

            pooled_h = torch.zeros(64, 140)
            # get_logger().info("len of nb_count: " + str(len(context['nb_counts'][0])))
            for i in range(len(context['nb_counts'][0])):
                num_neighbors = context['nb_counts'][0][i]
                sum_features = torch.sum(context['features'][0:num_neighbors])

                # torch.cat((pooled_h, torch.sum(context['features'][0:num_neighbors])), 0)

            # pooled_h = torch.spmm(graph_pool, h)
            score_over_layer += F.dropout(self.linears_prediction[layer](pooled_h), self.final_dropout, training = self.training)


        get_logger().info("Score Over Layer: " + str(score_over_layer))
        criterion = nn.CrossEntropyLoss()

        labels = context['nb_counts'][0]

        loss = criterion(score_over_layer, labels)
        return loss, score_over_layer, labels
    
    def metric_name(self):
        """Metric used for model evaluation."""
        return self.metric.name()

    def compute_metric(self, preds, labels):
        """Stub for metric evaluation."""
        if self.metric is not None:
            preds = torch.unsqueeze(torch.cat(preds, 0), 1)
            labels = torch.unsqueeze(torch.cat(labels, 0), 1).type(preds.dtype)
            return self.metric.compute(preds, labels)
        return torch.tensor(0.0)

    def query(self, graph: Graph, inputs: np.array):
        """Fetch training data from graph."""
        context = {"inputs": inputs}

        get_logger().info("Finished fetching inputs in query!")

        # context["label"] = graph.neighbor_count(
        #     nodes = context["inputs"],
        #     edge_types = np.ndarray(0),
        # )



        context['neighbors'] = graph.neighbors(
            nodes = context['inputs'],
            edge_types = np.array(0),
        )[0]

        get_logger().info(str(type(context['neighbors'][0])))

        context['nb_counts'] = graph.neighbors(
            nodes = context['inputs'],
            edge_types = np.array(0),
        )[3].astype(int)

        get_logger().info(str(type(context['nb_counts'][0])))



        get_logger().info("Finished fetching neighbors in query!")

        context["features"] = graph.node_features(
            context["neighbors"],
            np.array([[self.label_idx, self.label_dim]]),
            FeatureType.INT64,
        )

        get_logger().info("finished fetching node features in query!")

        get_logger().info('Context: ' + str(context))
        return context

    

