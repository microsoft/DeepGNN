# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import tensorflow as tf
import os, datetime, time
import glob


class DistributedSync:
    SYNC_FILE = "sync"

    def __init__(self, folder, task_index, num_tasks):
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

    def sync(self, tag):
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
