from .base import Layer
import tensorflow as tf


class AttentionHeader(Layer):
    "Attention header for GAT, reference: https://github.com/PetarV-/GAT/blob/master/utils/layers.py"

    def __init__(self, out_size, act=None):
        super().__init__()
        self.out_dim = out_size
        self.act = act

    def build(self, input_shape):
        self.conv0 = tf.keras.layers.Conv1D(self.out_dim, 1, use_bias=False)
        self.conv1 = tf.keras.layers.Conv1D(1, 1)
        self.conv2 = tf.keras.layers.Conv1D(1, 1)
        self.bias = tf.compat.v1.get_variable(
            "att.bias",
            shape=[self.out_dim],
            initializer=tf.compat.v1.zeros_initializer(),
        )
        self.built = True

    def call(self, seq):
        seq_fts = self.conv0(seq)
        f_1 = self.conv1(seq_fts)
        f_2 = self.conv2(seq_fts)
        logits = f_1 + tf.transpose(a=f_2, perm=[0, 2, 1])
        coefs = tf.nn.softmax(tf.nn.leaky_relu(logits))
        vals = tf.matmul(coefs, seq_fts)
        tf.compat.v1.logging.info(
            "AttentionHeader Tensor Shape: seq_flts {0}, f_1 {1}, f_2 {2}, logits {3}, coefs {4}, vals {5}".format(
                seq_fts.shape,
                f_1.shape,
                f_2.shape,
                logits.shape,
                coefs.shape,
                vals.shape,
            )
        )
        ret = tf.nn.bias_add(vals, self.bias)
        if self.act is None:
            return ret
        else:
            return self.act(ret)
