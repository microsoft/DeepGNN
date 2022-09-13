# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Hooks for distributed training."""
from typing import List
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
        checkpoint_dir: str,
        save_secs: int = None,
        save_steps: int = None,
        saver: object = None,
        checkpoint_basename: str = "model.ckpt",
        scaffold: object = None,
        listeners: List[object] = None,
    ):
        """Initialize hook."""
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
        """End session."""
        self.dist_sync.sync("session")
        super().end(session)

    # WORKAROUND: in windows we need to create this folder otherwise we will get this error:
    # Failed to create a NewWriteableFile
    def _save(self, session, step):
        if sys.platform == "win32":
            path = f"{self._save_path}-{step}_temp"
            os.makedirs(path, exist_ok=True)
        super()._save(session, step)


class SessionExitHook(tf.estimator.SessionRunHook):
    """Synchronize training at exit."""

    def __init__(self, dist_sync: DistributedSync):
        """Create lock."""
        self.dist_sync = dist_sync

    def end(self, session):
        """Wait for all workers to finish."""
        self.dist_sync.sync("session")
