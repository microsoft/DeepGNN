# Copyright 2018 Alibaba Group Holding Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Base classes to use with TF models."""
import collections
from typing import Dict, Any, Optional

import tensorflow as tf

from tensorflow.python.util import nest


_LAYER_UIDS: Dict[str, Any] = collections.defaultdict(lambda: 0)


def _get_layer_uid(layer_name: str = ""):
    _LAYER_UIDS[layer_name] += 1
    return _LAYER_UIDS[layer_name]


class Layer(object):
    """Layer class modeled after Keras (http://keras.io)."""

    def __init__(self, name=None, **kwargs):
        """Initialize layer."""
        self.built = False

        if name is None:
            layer_name = self.__class__.__name__.lower()
            name = layer_name + "_" + str(_get_layer_uid(layer_name))

        self._name = name
        self.partitioner = kwargs.get("partitioner", None)
        self.log_variable_info = []

    def build(self, input_shape: tuple):
        """Freeze layer."""
        self.built = True

    def call(self, inputs: tf.Tensor):
        """Return inputs to make derived classes simpler."""
        return inputs

    def __call__(self, inputs: tf.Tensor):
        """Freeze model and evaluate it."""
        input_shapes: Optional[tuple] = None
        if all(hasattr(x, "shape") for x in nest.flatten(inputs)):
            input_shapes = nest.map_structure(lambda x: x.shape, inputs)

        with tf.compat.v1.variable_scope(self._name):
            if not self.built:
                self.build(input_shapes)  # type: ignore
            outputs = self.call(inputs)
            return outputs

    def compute_output_shape(self, input_shape: tuple):
        """To be overriden in derived classes."""
        raise NotImplementedError()

    def print_model_variables(self):
        """Print debug information about model."""
        print("-------------Model internal variable---------------")
        [print("\t{0}\t{1}".format(k, v)) for k, v in self.log_variable_info]
        print("-------------Local variables-----------------------")
        [
            print("\t".join(["", str(v.dtype), str(v.get_shape()), v.name]))
            for v in tf.compat.v1.local_variables()
        ]
        print("-------------Global variables----------------------")
        [
            print("\t".join(["", str(v.dtype), str(v.get_shape()), v.name]))
            for v in tf.compat.v1.global_variables()
        ]
        print("-------------Trainable Variables-------------------")
        [
            print("\t".join(["", str(v.dtype), str(v.get_shape()), v.name]))
            for v in tf.compat.v1.trainable_variables()
        ]
        import sys

        sys.stdout.flush()


class Dense(Layer):
    """Basic full-connected layer."""

    def __init__(
        self,
        dim,
        activation=None,
        use_bias=True,
        kernel_initializer=lambda: tf.compat.v1.keras.initializers.VarianceScaling(
            scale=0.36, distribution="uniform"
        ),
        bias_initializer=lambda: tf.compat.v1.constant_initializer(value=0.0002),
        **kwargs
    ):
        """Initialize dense layer."""
        super(Dense, self).__init__(**kwargs)
        self.dim = dim
        self.activation = activation
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer

    def build(self, input_shape):
        """Compute shape, bias and then freeze layer."""
        input_shape = tf.TensorShape(input_shape)
        self.kernel = tf.compat.v1.get_variable(
            "kernel",
            shape=[input_shape[-1], self.dim],
            initializer=self.kernel_initializer(),
        )
        if self.use_bias:
            self.bias = tf.compat.v1.get_variable(
                "bias", shape=[self.dim], initializer=self.bias_initializer()
            )
        else:
            self.bias = None
        self.built = True

    def call(self, inputs):
        """Evaluate layer."""
        rank = inputs.shape.ndims
        if rank > 2:
            outputs = tf.tensordot(inputs, self.kernel, [[rank - 1], [0]])
        else:
            outputs = tf.matmul(inputs, self.kernel)
        if self.use_bias:
            outputs = tf.nn.bias_add(outputs, self.bias)
        if self.activation:
            outputs = self.activation(outputs)
        return outputs


class Embedding(Layer):
    """Id to dense vector embedding."""

    def __init__(
        self,
        max_id,
        dim,
        initializer=lambda: tf.compat.v1.truncated_normal_initializer(stddev=0.1),
        **kwargs
    ):
        """Initialize embedding layer."""
        super(Embedding, self).__init__(**kwargs)
        self.max_id = max_id
        self.dim = dim
        self.initializer = initializer

    def build(self, input_shape):
        """Compute layer shape and freeze it."""
        self.embeddings = tf.compat.v1.get_variable(
            "embeddings",
            shape=[self.max_id + 1, self.dim],
            initializer=self.initializer(),
            partitioner=self.partitioner,
        )
        self.built = True

    def call(self, inputs):
        """Lookup embeddings for inputs."""
        shape = inputs.shape
        inputs = tf.reshape(inputs, [-1])
        output_shape = shape.concatenate(self.dim)
        output_shape = [d if d is not None else -1 for d in output_shape.as_list()]
        return tf.reshape(
            tf.nn.embedding_lookup(params=self.embeddings, ids=inputs), output_shape
        )


class SparseEmbedding(Embedding):
    """Sparse id to dense vector embedding."""

    def __init__(
        self,
        max_id,
        dim,
        initializer=lambda: tf.compat.v1.truncated_normal_initializer(stddev=0.0002),
        combiner="sum",
        **kwargs
    ):
        """Initialize sparse embedding layer."""
        super(SparseEmbedding, self).__init__(
            max_id=max_id, dim=dim, initializer=initializer, **kwargs
        )
        self.combiner = combiner

    def call(self, inputs):
        """Lookup embeddings for sparse inputs."""
        return tf.nn.embedding_lookup_sparse(
            params=self.embeddings,
            sp_ids=inputs,
            sp_weights=None,
            combiner=self.combiner,
        )
