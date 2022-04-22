# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import tensorflow as tf
from deepgnn.tf.common.dist_sync import DistributedSync


class SessionExitHook(tf.estimator.SessionRunHook):
    def __init__(self, dist_sync: DistributedSync):
        self.dist_sync = dist_sync

    def end(self, session):
        self.dist_sync.sync("session")
