# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import glob
import numpy as np
import os
import tensorflow as tf

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from deepgnn import get_logger


class TestHelper:
    @staticmethod
    def get_tf2_summary_value(summary_dir, metric_name):

        events_file_pattern = os.path.join(summary_dir, "events*")
        events_files = sorted(glob.glob(events_file_pattern))
        get_logger().info(f"event files: {events_files}")
        events = EventAccumulator(summary_dir)
        events.Reload()

        metric_values = []
        for w, s, t in events.Tensors(metric_name):
            val = tf.make_ndarray(t)
            metric_values.append(np.asscalar(val))
        return metric_values
