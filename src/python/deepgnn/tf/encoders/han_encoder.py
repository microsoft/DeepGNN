# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Encoders for HAN model."""
import numpy as np
import tensorflow as tf

from deepgnn.tf import layers

from .att_encoder import AttEncoder

from typing import Tuple, Union


class SimpleAttLayer(tf.keras.layers.Layer):
    """Reference: https://github.com/Jhy1993/HAN/blob/master/utils/layers.py#L132."""

    def __init__(self, attention_size: int):
        """Initialize encoder."""
        super().__init__()
        self.att_size = attention_size

    def build(self, input_shape: Tuple[int, int, int]):
        """Create internal variables."""
        hidden_size = input_shape[2].value  # D value - hidden size of the RNN layer
        # Trainable parameters
        # fmt:off
        self.w_omega = tf.Variable(tf.random.normal([hidden_size, self.att_size], stddev=0.1), name="w_omega")
        self.b_omega = tf.Variable(tf.random.normal([self.att_size], stddev=0.1), name="b_omega")
        self.u_omega = tf.Variable(tf.random.normal([self.att_size], stddev=0.1), name="u_omega")
        # fmt:on

    def call(
        self, inputs, time_major: bool = False, return_alphas: bool = False
    ) -> Union[tf.Tensor, Tuple[tf.Tensor, tf.Tensor]]:
        """Compute embeddings."""
        if time_major:
            # (T,B,D) => (B,T,D)
            inputs = tf.array_ops.transpose(inputs, [1, 0, 2])

        with tf.name_scope("v"):
            # Applying fully connected layer with non-linear activation to each of the B*T timestamps;
            #  the shape of `v` is (B,T,D)*(D,A)=(B,T,A), where A=attention_size
            v = tf.tanh(tf.tensordot(inputs, self.w_omega, axes=1) + self.b_omega)

        # For each of the timestamps its vector of size A from `v` is reduced with `u` vector
        vu = tf.tensordot(v, self.u_omega, axes=1, name="vu")  # (B,T) shape
        alphas = tf.nn.softmax(vu, name="alphas")  # (B,T) shape

        # Output of (Bi-)RNN is reduced with attention vector; the result has (B,D) shape
        output = tf.reduce_sum(input_tensor=inputs * tf.expand_dims(alphas, -1), axis=1)

        if not return_alphas:
            return output
        else:
            return output, alphas


class HANEncoder(AttEncoder):
    """HAN Encoder (https://arxiv.org/pdf/1903.07293)."""

    def __init__(
        self,
        edge_types: list,
        head_num: list,
        hidden_dim: list,
        nb_num: list,
        feature_idx: int = -1,
        feature_dim: int = 0,
        max_id: int = -1,
        out_dim: int = 128,
        **kwargs
    ):
        """Initialize encoder."""
        if len(head_num) == 0:
            raise ValueError("head_num can't be empty.")
        if len(hidden_dim) == 0:
            raise ValueError("hidden_dim can't be empty.")
        if len(head_num) != len(hidden_dim):
            raise ValueError(
                "length of head_num must be equal with length of hidden_dim."
            )

        super(HANEncoder, self).__init__(
            edge_types[0],
            feature_idx,
            feature_dim,
            head_num[0],
            nb_num[0],
            out_dim,
            **kwargs
        )
        self.feature_idx = np.array([[feature_idx, feature_dim]], dtype=np.int32)
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.head_num = head_num
        self.nb_num = nb_num
        self.edge_types = edge_types
        self.max_id = max_id

        self.metapath_num = int(len(self.edge_types) / len(self.nb_num))
        self.simple_att_layer = SimpleAttLayer(out_dim)

        self.att_headers = [[[]]] * self.metapath_num  # type: ignore
        for i in range(self.metapath_num):
            for layer_id in range(len(self.head_num)):
                for j in range(self.head_num[layer_id]):
                    self.att_headers[i][layer_id].append(
                        layers.AttentionHeader(self.hidden_dim[layer_id], act=tf.nn.elu)
                    )

    def _multi_head_layer(
        self, inputs: tf.Tensor, metapath_id: int, layer_id: int
    ) -> tf.Tensor:
        hidden = []
        for i in range(0, self.head_num[layer_id]):
            hidden_val = self.att_headers[metapath_id][layer_id][i](inputs)
            tf.compat.v1.logging.info(
                "layer {0} hidden shape {1}".format(layer_id, hidden_val.shape)
            )
            hidden_val = tf.reshape(
                hidden_val, [-1, sum(self.nb_num) + 1, self.hidden_dim[layer_id]]
            )
            hidden.append(hidden_val)
        return tf.concat(hidden, -1)

    def call(self, inputs: Tuple[np.ndarray, np.ndarray]) -> tf.Tensor:  # type: ignore
        """Compute embeddings."""
        node_feats_arr, neighbor_feats_arr = inputs
        embed_list = []
        total_nb_num = np.prod(self.nb_num)
        for i in range(self.metapath_num):
            seq = tf.concat([node_feats_arr[i], neighbor_feats_arr[i]], 1)

            h_1 = self._multi_head_layer(seq, metapath_id=i, layer_id=0)
            for j in range(1, len(self.head_num)):
                h_1 = self._multi_head_layer(h_1, metapath_id=i, layer_id=j)

            output_dim = self.hidden_dim[-1] * self.head_num[-1]
            out = tf.reshape(h_1, [-1, total_nb_num + 1, output_dim])
            out = tf.slice(out, [0, 0, 0], [-1, 1, output_dim])

            tf.compat.v1.logging.info("out shape {0}".format(out.shape))
            out = tf.reshape(out, [-1, output_dim])
            embed_list.append(tf.expand_dims(out, axis=1))

        multi_embed = tf.concat(embed_list, axis=1)
        final_embed, _ = self.simple_att_layer(
            multi_embed, time_major=False, return_alphas=True
        )
        return final_embed
