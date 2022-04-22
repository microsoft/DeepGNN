# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import tensorflow as tf
import os
import sys
from deepgnn.tf.common.dist_sync import DistributedSync


class ChiefCheckpointSaverHook(tf.estimator.CheckpointSaverHook):
    """
    Chief only hooks, chief session will call `dist_sync.sync()` to wait all workers session.
    Once all workers finished, chief session run `end(session)` to save the final checkpoint.
    """

    def __init__(
        self,
        dist_sync: DistributedSync,
        checkpoint_dir,
        save_secs=None,
        save_steps=None,
        saver=None,
        checkpoint_basename="model.ckpt",
        scaffold=None,
        listeners=None,
    ):
        super().__init__(
            checkpoint_dir,
            save_secs,
            save_steps,
            saver,
            checkpoint_basename,
            scaffold,
            listeners,
        )
        assert dist_sync.task_index == 0
        self.dist_sync = dist_sync

    def end(self, session):
        self.dist_sync.sync("session")
        super().end(session)

    # WORKAROUND: in windows we need to create this folder otherwise we will get this error:
    # Failed to create a NewWriteableFile
    def _save(self, session, step):
        if sys.platform == "win32":
            path = f"{self._save_path}-{step}_temp"
            os.makedirs(path, exist_ok=True)
        super()._save(session, step)
