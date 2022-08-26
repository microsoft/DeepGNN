# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Graphsage models to solve link prediction problem."""
import numpy as np
import tensorflow as tf

from deepgnn.tf.nn import sage_conv
from deepgnn import get_logger
from typing import List, Optional

from deepgnn.graph_engine import Graph
from sage import SAGEQuery, SAGEQueryParameter  # type: ignore


class SAGELinkPredictionQuery(SAGEQuery):
    """GraphSAGE query for edges."""

    def __init__(self, param: SAGEQueryParameter):
        """Initialize query."""
        super().__init__(param)

    def query_training(
        self, graph: Graph, inputs: np.ndarray, return_shape: bool = False
    ) -> tuple:
        """Fetch data to train model."""
        # fmt: off
        seed_edges = inputs  # [Batch_size, 3]
        src_nodes, dst_nodes = seed_edges[:, 0], seed_edges[:, 1]
        src_graph_nodes, src_neighbor_list_idx = self._query_neighbor(graph, src_nodes)
        dst_graph_nodes, dst_neighbor_list_idx = self._query_neighbor(graph, dst_nodes)

        edge_label = graph.edge_features(seed_edges, self.label_meta, self.param.label_type)

        graph_tensor = tuple([src_graph_nodes, dst_graph_nodes, edge_label] + src_neighbor_list_idx + dst_neighbor_list_idx)

        if return_shape:
            # N is the number of `nodes`, which is variable because `inputs` nodes are different.
            N, M = None, None
            shapes = [
                [N],                                        # src_graph_nodes
                [M],                                        # dst_graph_nodes
                [inputs.shape[0], self.param.label_dim],    # edge_label
            ]

            nb_shapes = [list(x.shape) for x in src_neighbor_list_idx]
            shapes.extend(nb_shapes)
            nb_shapes = [list(x.shape) for x in dst_neighbor_list_idx]
            shapes.extend(nb_shapes)
            return graph_tensor, shapes

        return graph_tensor
        # fmt: on

    def query_inference(
        self, graph: Graph, inputs: np.ndarray, return_shape: bool = False
    ) -> tuple:
        """Inference `inputs` is different with train `inputs`.

        * for training job, `inputs` are edges.
        * for inference job, `inputs` are nodes.
        """
        # fmt: off
        seed_nodes = inputs
        graph_nodes, neighbor_list_idx = self._query_neighbor(graph, seed_nodes)

        graph_tensor = tuple([graph_nodes] + neighbor_list_idx)

        if return_shape:
            # N is the number of `nodes`, which is variable because `inputs` nodes are different.
            N = None
            shapes = [
                [N]  # graph_nodes
            ]

            nb_shapes = [list(x.shape) for x in neighbor_list_idx]
            shapes.extend(nb_shapes)
            return graph_tensor, shapes

        return graph_tensor
        # fmt: on


class GraphSAGELinkPrediction(tf.keras.Model):
    """GraphSAGE model for link predcition."""

    def __init__(
        self,
        in_dim,
        layer_dims: List[int],
        num_classes: int,
        num_samples: List[int],
        dropout: float = 0.0,
        agg_type: str = "mean",
        identity_embed_shape: List[int] = [],
        concat: bool = True,
    ):
        """Initialize model."""
        super().__init__()
        self.logger = get_logger()
        # fmt: off
        assert len(layer_dims) == len(num_samples), f"layer_dim {layer_dims}, num_samplers {num_samples}"
        # fmt: on

        self.dims = [in_dim] + layer_dims
        self.num_classes = num_classes
        self.num_samples = num_samples
        self.dropout = dropout
        self.identity_embed_shape = identity_embed_shape
        self.concat = concat
        self.inference_node: Optional[str] = None

        # fmt: off
        # init src|dst aggregate layer.
        self.src_aggs = sage_conv.init_aggregators(agg_type, layer_dims, dropout, self.concat)
        self.dst_aggs = sage_conv.init_aggregators(agg_type, layer_dims, dropout, self.concat)
        # fmt: on

        # use identity features
        if self.identity_embed_shape is not None:
            self.max_id = identity_embed_shape[0] - 1
            with tf.name_scope("op"):
                self.node_emb = tf.Variable(
                    tf.initializers.GlorotUniform()(shape=self.identity_embed_shape),
                    name="node_emb",
                    dtype=tf.float32,
                    trainable=True,
                )

        self.auc = tf.keras.metrics.AUC(name="auc")

    def set_inference_node(self, node: str):
        """Fix inference node."""
        assert node in ["src", "dst"]
        self.inference_node = node

    def inference(self, inputs):
        """Calculate embeddings."""
        nodes = inputs[0]
        neighbor_list = inputs[1:]
        valid_nodes = tf.where(nodes >= 0, nodes, tf.ones_like(nodes) * self.max_id)
        feat = tf.nn.embedding_lookup(self.node_emb, tf.reshape(valid_nodes, [-1]))

        if self.inference_node == "src":
            emb = self._sage_embeding(feat, neighbor_list, self.src_aggs)
        else:
            emb = self._sage_embeding(feat, neighbor_list, self.dst_aggs)

        input_nodes_idx = neighbor_list[0]
        self.node_ids = tf.nn.embedding_lookup(nodes, input_nodes_idx)
        self.out_emb = emb

        return self.node_ids, self.out_emb

    def _sage_embeding(self, feat, neighbor_list, agg_layer):
        hidden = [tf.nn.embedding_lookup(feat, nb) for nb in neighbor_list]
        output = sage_conv.aggregate(
            hidden, agg_layer, self.num_samples, self.dims, self.concat
        )
        output = tf.nn.l2_normalize(output, 1)
        if self.dropout != 0.0:
            output = tf.nn.dropout(output, rate=self.dropout)
        return output

    def call(self, inputs, training=True):
        """Compute predictions, loss and AUC metric."""
        if not training and self.inference_node is not None:
            return self.inference(inputs)

        # fmt: off
        src_graph_nodes, dst_graph_nodes, labels = inputs[0:3]
        nb_list_size = len(self.num_samples) + 1
        src_neighbor_list = inputs[3 : 3 + nb_list_size]
        dst_neighbor_list = inputs[3 + nb_list_size :]

        valid_src = tf.where(src_graph_nodes >= 0, src_graph_nodes, tf.ones_like(src_graph_nodes) * self.max_id)
        src_feat = tf.nn.embedding_lookup(self.node_emb, tf.reshape(valid_src, [-1]))
        valid_dst = tf.where(dst_graph_nodes >= 0, dst_graph_nodes, tf.ones_like(dst_graph_nodes) * self.max_id)
        dst_feat = tf.nn.embedding_lookup(self.node_emb, tf.reshape(valid_dst, [-1]))
        # fmt: on

        src_output = self._sage_embeding(src_feat, src_neighbor_list, self.src_aggs)
        dst_output = self._sage_embeding(dst_feat, dst_neighbor_list, self.dst_aggs)
        self.logger.info(
            f"src|dst output shape: {src_output.shape}, {dst_output.shape}"
        )

        # loss
        logits = tf.reduce_sum(input_tensor=src_output * dst_output, axis=-1)
        logits = tf.reshape(logits, [-1, 1])
        labels = tf.reshape(labels, [-1, 1])
        entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels)
        loss = tf.reduce_mean(input_tensor=entropy)
        predictions = tf.sigmoid(logits)

        # update metrics
        self.auc.update_state(labels, predictions)

        return predictions, loss, {"auc": self.auc.result()}

    def train_step(self, data: dict):
        """Override base train_step."""
        with tf.GradientTape() as tape:
            _, loss, metrics = self(data, training=True)

        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        result = {"loss": loss}
        result.update(metrics)
        return result

    def test_step(self, data: dict):
        """Override base test_step."""
        _, loss, metrics = self(data, training=False)
        result = {"loss": loss}
        result.update(metrics)
        return result

    def predict_step(self, data: dict):
        """Override base predict_step."""
        self(data, training=False)
        return [self.node_ids, self.out_emb]
