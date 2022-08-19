# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Convolution layers for sage models."""

import tensorflow as tf
from typing import Callable
from typing import List, Optional


class MeanAggregator(tf.keras.layers.Layer):
    """
    Mean Aggregation for GraphSAGE.

    reference: https://github.com/williamleif/GraphSAGE/blob/a0fdef95dca7b456dab01cb35034717c8b6dd017/graphsage/aggregators.py#L6
    """

    def __init__(
        self,
        output_dim: int,
        dropout: float = 0.0,
        bias: bool = False,
        act: Optional[Callable] = tf.nn.relu,
        concat: bool = False,
    ):
        """Initialize aggregator."""
        super().__init__()
        self.output_dim = output_dim
        self.dropout = dropout
        self.enable_bias = bias
        self.act = act
        self.concat = concat

        # build
        self.neigh_weight = tf.keras.layers.Dense(
            self.output_dim, use_bias=False, name="neigh_weights"
        )
        self.self_weight = tf.keras.layers.Dense(
            self.output_dim, use_bias=False, name="self_weights"
        )
        if self.enable_bias:
            self.bias = self.add_weight(
                name="bias",
                shape=[self.out_dim],
                initializer="zeros",
                dtype=tf.float32,
                trainable=True,
            )

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Evaluate aggregator."""
        self_vecs, neig_vecs = inputs

        if self.dropout != 0.0:
            self_vecs = tf.nn.dropout(self_vecs, rate=self.dropout)  # [N, in_dim]
            neig_vecs = tf.nn.dropout(
                neig_vecs, rate=self.dropout
            )  # [N, num_nb, in_dim]
        neig_mean = tf.reduce_mean(neig_vecs, axis=1)

        from_neig = self.neigh_weight(neig_mean)  # [N, output_dim]
        from_self = self.self_weight(self_vecs)  # [N, output_dim]

        if self.concat:
            output = tf.concat([from_self, from_neig], axis=1)
        else:
            output = tf.add_n([from_self, from_neig])

        if self.enable_bias:
            output += self.bias

        if self.act:
            output = self.act(output)
        return output


class MaxPoolingAggregator(tf.keras.layers.Layer):
    """
    Max-pooling Aggregation for GraphSAGE.

    reference: https://github.com/williamleif/GraphSAGE/blob/a0fdef95dca7b456dab01cb35034717c8b6dd017/graphsage/aggregators.py#L119
    """

    def __init__(
        self,
        output_dim: int,
        dropout: float = 0.0,
        bias: bool = False,
        act: Callable = tf.nn.relu,
        concat: bool = False,
        hidden_dim: int = 512,
    ):
        """Initialize aggregator."""
        super().__init__()
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.enable_bias = bias
        self.act = act
        self.concat = concat

        self.mlp_layer = tf.keras.layers.Dense(
            self.hidden_dim, activation=tf.nn.relu, name="mlp_layer"
        )
        self.neigh_weight = tf.keras.layers.Dense(
            self.output_dim, use_bias=False, name="neigh_weights"
        )
        self.self_weight = tf.keras.layers.Dense(
            self.output_dim, use_bias=False, name="self_weights"
        )
        if self.enable_bias:
            self.bias = self.add_weight(
                name="bias",
                shape=[self.out_dim],
                initializer="zeros",
                dtype=tf.float32,
                trainable=True,
            )

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Evaluate aggregator."""
        self_vecs, neig_vecs = inputs  # [N, in_dim], [N, num_nb, in_dim]
        neigh_h = neig_vecs
        if self.dropout != 0.0:
            neigh_h = tf.nn.dropout(neigh_h, rate=self.dropout)
        neigh_h = self.mlp_layer(neigh_h)
        neigh_h = tf.reduce_max(neigh_h, axis=1)

        from_neig = self.neigh_weight(neigh_h)  # [N, output_dim]
        from_self = self.self_weight(self_vecs)  # [N, output_dim]

        if self.concat:
            output = tf.concat([from_self, from_neig], axis=1)
        else:
            output = tf.add_n([from_self, from_neig])

        if self.enable_bias:
            output += self.bias

        if self.act:
            output = self.act(output)
        return output


class LSTMAggregator(tf.keras.layers.Layer):
    """
    LSTM Aggregation for GraphSAGE.

    Reference: https://github.com/williamleif/GraphSAGE/blob/a0fdef95dca7b456dab01cb35034717c8b6dd017/graphsage/aggregators.py#L363
    """

    def __init__(
        self,
        output_dim: int,
        dropout: float = 0.0,
        bias: bool = False,
        act: Callable = tf.nn.relu,
        concat: bool = False,
        hidden_dim: int = 128,
    ):
        """Initialize aggregator."""
        super().__init__()
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.enable_bias = bias
        self.act = act
        self.concat = concat

        self.neigh_weight = tf.keras.layers.Dense(
            self.output_dim, use_bias=False, name="neigh_weights"
        )
        self.self_weight = tf.keras.layers.Dense(
            self.output_dim, use_bias=False, name="self_weights"
        )
        if self.enable_bias:
            self.bias = self.add_weight(
                name="bias",
                shape=[self.out_dim],
                initializer="zeros",
                dtype=tf.float32,
                trainable=True,
            )
        self.lstm_cell = tf.keras.layers.LSTMCell(self.hidden_dim)
        self.rnn = tf.keras.layers.RNN(self.lstm_cell, return_sequences=True)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Evaluate aggregator."""
        self_vecs, neig_vecs = inputs  # [N, in_dim], [N, num_nb, in_dim]

        used = tf.sign(tf.reduce_max(tf.abs(neig_vecs), axis=2))
        length = tf.reduce_sum(used, axis=1)
        length = tf.maximum(length, tf.constant(1.0))
        length = tf.cast(length, tf.int32)

        rnn_outputs = self.rnn(neig_vecs)  # , sequence_length=length)
        max_len = tf.shape(rnn_outputs)[1]
        out_size = int(rnn_outputs.get_shape()[2])
        index = tf.range(0, tf.shape(rnn_outputs)[0]) * max_len + (length - 1)
        flat = tf.reshape(rnn_outputs, [-1, out_size])
        neigh_h = tf.gather(flat, index)

        from_neig = self.neigh_weight(neigh_h)  # [N, output_dim]
        from_self = self.self_weight(self_vecs)  # [N, output_dim]

        if self.concat:
            output = tf.concat([from_self, from_neig], axis=1)
        else:
            output = tf.add_n([from_self, from_neig])

        if self.enable_bias:
            output += self.vars["bias"]
        if self.act is not None:
            output = self.act(output)
        return output


_AGG_CLASS = {
    "mean": MeanAggregator,
    "maxpool": MaxPoolingAggregator,
    "lstm": LSTMAggregator,
}


def init_aggregators(
    agg_type: str, layer_dims: List[int], dropout: float = 0.0, concat: bool = True
):
    """Initialize aggregators based on type."""
    assert agg_type in _AGG_CLASS, f"unknown agg_type: {agg_type}"
    _agg_class = _AGG_CLASS[agg_type]

    agg_layers = []
    N = len(layer_dims)
    for layer in range(N):
        act = None if layer == N - 1 else tf.nn.relu
        agg = _agg_class(
            output_dim=layer_dims[layer], dropout=dropout, act=act, concat=concat
        )
        agg_layers.append(agg)
    return agg_layers


def aggregate(
    hidden: tf.Tensor,
    agg_layers: List[tf.keras.layers.Layer],
    num_samples: List[int],
    dims: tuple,
    concat: bool = True,
) -> tf.Tensor:
    """SAGE aggregation."""
    N = len(num_samples)
    for layer in range(N):
        aggregator = agg_layers[layer]
        next_hidden = []
        dim_mult = 2 if concat and (layer != 0) else 1
        for hop in range(N - layer):
            neigh_shape = [-1, num_samples[N - hop - 1], dim_mult * dims[layer]]
            h = aggregator((hidden[hop], tf.reshape(hidden[hop + 1], neigh_shape)))
            next_hidden.append(h)
        hidden = next_hidden
    assert len(hidden) == 1, hidden
    output = hidden[0]
    return output
