# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import tensorflow as tf


def masked_softmax_cross_entropy(
    preds: tf.Tensor, labels: tf.Tensor, mask: tf.Tensor = None
):
    """Softmax cross-entropy loss with masking."""
    if mask is not None:
        preds = preds[mask]
        labels = labels[mask]
    loss = tf.nn.softmax_cross_entropy_with_logits(
        logits=preds, labels=tf.stop_gradient(labels)
    )
    return tf.reduce_mean(loss)


def masked_accuracy(
    preds: tf.Tensor, labels: tf.Tensor, mask: tf.Tensor = None, dtype=tf.float32
):
    """Accuracy with masking."""
    if mask is not None:
        preds = preds[mask]
        labels = labels[mask]
    correct_prediction = tf.equal(tf.argmax(preds, 1), tf.argmax(labels, 1))
    accuracy_all = tf.cast(correct_prediction, dtype)
    return tf.reduce_mean(accuracy_all)
