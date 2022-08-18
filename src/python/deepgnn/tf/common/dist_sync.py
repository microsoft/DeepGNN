# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Synchronization for distributed training."""
import tensorflow as tf
import os
import datetime
import time
import glob


class DistributedSync:
    """Distributed synchronization with plain files."""

    SYNC_FILE = "sync"

    def __init__(self, folder: str, task_index: int, num_tasks: int):
        """
        Initialize folder.

        Task with index 0 must be called first to start from a fresh state.
        """
        self.folder = folder
        self.task_index = task_index
        self.num_tasks = num_tasks
        if self.task_index == 0:
            self._cleanup_sync_files()

    def _cleanup_sync_files(self):
        retry = 100
        filelist = glob.glob(os.path.join(self.folder, "{}.*".format(self.SYNC_FILE)))
        for fname in filelist:
            while os.path.exists(fname) and retry > 0:
                try:
                    tf.compat.v1.logging.info("remove {}".format(fname))
                    os.remove(fname)
                except FileNotFoundError as err:
                    tf.compat.v1.logging.info(
                        "Oops! Delete file ({0}). OSError: {1}".format(fname, err)
                    )
                    time.sleep(60)
                    retry -= 1

    def sync(self, tag: str):
        """Block until all workers are ready."""
        with open(
            os.path.join(
                self.folder, "{}.{}.{}".format(self.SYNC_FILE, tag, self.task_index)
            ),
            "w",
        ) as w:
            w.write(str(datetime.datetime.now()))
        for i in range(self.num_tasks):
            while not os.path.exists(
                os.path.join(self.folder, "{}.{}.{}".format(self.SYNC_FILE, tag, i))
            ):
                time.sleep(30)
                tf.compat.v1.logging.info("worker {}-{} is not ready...".format(i, tag))
        tf.compat.v1.logging.info("all workers-{} are done.".format(tag))
