# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import re
import random
import glob
import threading
from typing import Optional, List, Iterator
from queue import Queue
from concurrent.futures import ThreadPoolExecutor
import xml.etree.ElementTree as et
from azure.datalake.store import core, lib
from deepgnn import get_logger
from deepgnn.graph_engine._base import get_fs

LINE_DELIMITER = int.from_bytes(b"\n", "little")


# A indicator which is put to the queue with the count.
# __next__ function in iterable will get each queue item
# till reach the end which is specified by this indicator.
class FetchDone(object):
    def __init__(self, count=0, exception=None):
        self.count = count
        self.exception_info = exception


class AdlCredentialParser:
    @staticmethod
    def read_credentials(config: str = None) -> dict:
        adl_config = {"TENANT_ID": "", "CLIENT_SECRET": "", "CLIENT_ID": ""}

        if config is not None and len(config) > 0:
            # if the config is a path, load it from file.
            if os.path.exists(config):
                tree = et.parse(config)
                root = tree.getroot()
            else:
                # if the config is xml content, load it from string.
                root = et.fromstring(config)
        else:
            hadoop_home = os.getenv("HADOOP_HOME", "/usr/local/hadoop")
            path = os.path.join(hadoop_home, "etc/hadoop/core-site.xml")
            tree = et.parse(path)
            root = tree.getroot()

        for item in root.findall("./property"):
            if item[0].text == "fs.adl.oauth2.client.id":
                adl_config["CLIENT_ID"] = item[1].text  # type: ignore
            if item[0].text == "fs.adl.oauth2.credential":
                adl_config["CLIENT_SECRET"] = item[1].text  # type: ignore
            if item[0].text == "fs.adl.oauth2.refresh.url":
                c = re.search(
                    "[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}",
                    item[1].text,  # type: ignore
                )
                if c is not None:
                    adl_config["TENANT_ID"] = c.group(0)

        return adl_config


class TextFileIterator:
    """Iterate lines from data lake gen1 files."""

    def __init__(
        self,
        filename: str,
        store_name: Optional[str] = None,
        adl_config: Optional[str] = None,
        batch_size: int = 512,
        epochs: int = 1,
        read_block_in_M: int = 50,
        buffer_queue_size: int = 3,
        thread_count: int = 10,
        worker_index: int = 0,
        num_workers: int = 1,
        shuffle: bool = False,
        drop_last: bool = False,
    ):
        if store_name is not None and len(store_name) != 0:
            spn = AdlCredentialParser.read_credentials(config=adl_config)

            principal_token = lib.auth(
                tenant_id=spn["TENANT_ID"],
                client_secret=spn["CLIENT_SECRET"],
                client_id=spn["CLIENT_ID"],
            )
            self.adl = core.AzureDLFileSystem(
                token=principal_token, store_name=store_name
            )

        self.fs, _ = get_fs(filename)

        # file index is the indicator of which file is now reading.
        self.file_idx = 0
        # the offset where the current file already read.
        self.offset = self.get_offset()
        # a state to show whether the iteration is finished.
        self.end = False
        self.batch_size = batch_size
        # the block size to read in each read request.
        self.length = int(read_block_in_M * 1000000)
        self.epochs = epochs
        self.left_epochs = epochs
        # a flag to terminate the reading thread.
        self.stop = False
        self.remain_lines: List[str] = []
        self.thread_count = thread_count
        if self.thread_count < 0:
            self.thread_count = 1
        self.remain_bytes = bytearray()
        # queue of file content bytes.
        self.outputs: Queue = Queue(buffer_queue_size)
        self.pool = ThreadPoolExecutor(1)
        self.workers = []  # type: ignore
        self.files = self.get_worker_files(filename, worker_index, num_workers)
        self.shuffle = shuffle
        self.drop_last = drop_last

        if shuffle:
            random.shuffle(self.files)

        get_logger().info(f"[{num_workers},{worker_index}] Input files: {self.files}")

    def reset(self):
        self.join()

        self.file_idx = 0
        self.offset = self.get_offset()
        self.left_epochs = self.epochs
        self.end = False
        self.stop = False
        self.remain_bytes = bytearray()
        self.remain_lines = []
        self.workers = []  # type: ignore

        self.pool = ThreadPoolExecutor(1)
        f = self.pool.submit(self._read_block)
        self.workers.append(f)

    def _read_block_by_threads(self) -> dict:
        decoded_buf: List[dict] = []
        for n in range(self.thread_count):
            decoded_buf.append(
                {"prefix": None, "lines": [], "suffix": None, "length": 0}
            )

        with ThreadPoolExecutor(max_workers=self.thread_count) as pool:

            def _read_block_worker(index, offset, length):
                # stop the pending workers to prevent read from closed files.
                if self.stop:
                    return

                if not hasattr(self, "adl"):
                    with self.fs.open(self.files[self.file_idx], "rb") as f:
                        f.seek(offset)
                        sub_block = f.read(length)
                else:
                    sub_block = self.adl.read_block(
                        self.files[self.file_idx], offset, length
                    )

                decoded_buf[index]["length"] = len(sub_block)
                if len(sub_block) == 0:
                    return

                try:
                    first_delim_index = sub_block.index(b"\n")
                except ValueError:
                    decoded_buf[index]["prefix"] = bytearray(sub_block)
                    return
                else:
                    decoded_buf[index]["prefix"] = bytearray(
                        sub_block[0:first_delim_index]
                    )

                last_delim_index = len(sub_block) - 1 - sub_block[::-1].index(b"\n")
                decoded_buf[index]["suffix"] = bytearray(
                    sub_block[last_delim_index + 1 :]
                )
                if last_delim_index != first_delim_index:
                    decoded_buf[index]["lines"] = (
                        sub_block[first_delim_index + 1 : last_delim_index]
                        .decode("utf-8")
                        .splitlines()
                    )

            len_to_read = self.get_length_to_read()
            sub_len = len_to_read // self.thread_count
            last_id = self.thread_count - 1
            for thread_idx in range(last_id):
                pool.submit(
                    _read_block_worker,
                    thread_idx,
                    self.offset + (thread_idx * sub_len),
                    sub_len,
                )

            pool.submit(
                _read_block_worker,
                last_id,
                self.offset + (last_id * sub_len),
                len_to_read - last_id * sub_len,
            )

        # merge the prefix and suffix between blocks.
        # first we get the head block of the block list,
        # then we will concat each block after the head into head block.
        head_block = decoded_buf[0]
        decoded_buf.remove(head_block)

        while len(decoded_buf) > 0:
            # get current block
            cur_buf = decoded_buf[0]
            decoded_buf.remove(cur_buf)

            # calculate total read size.
            head_block["length"] += cur_buf["length"]
            if cur_buf["prefix"] is None:
                continue

            head_buf = (
                head_block["suffix"]
                if head_block["suffix"] is not None
                else head_block["prefix"]
            )

            if head_buf is None:
                head_block["prefix"] = cur_buf["prefix"]
                head_buf = head_block["prefix"]
            else:
                head_buf += cur_buf["prefix"]
            # found '\n' in current block
            if cur_buf["suffix"] is not None:
                if head_block["suffix"] is not None:
                    # there is '\n' in head block, concat the current prefix and head's suffix and decode.
                    head_block["lines"].extend(
                        head_buf.decode("utf-8").splitlines(False)
                    )

                # concat lines in current block into head block
                head_block["lines"].extend(cur_buf["lines"])
                head_block["suffix"] = cur_buf["suffix"]

        return head_block

    def get_length_to_read(self) -> int:
        return self.length

    def get_offset(self) -> int:
        return 0

    def _read_block(self):
        while not self.stop:
            # try to read some data from current file using multi-threading
            block = self._read_block_by_threads()
            total_len = block["length"]

            # if read empty, it means end of the file, then we need to read next file
            while total_len == 0 and self.file_idx < (len(self.files) - 1):
                self.file_idx += 1
                self.offset = self.get_offset()
                self.outputs.put(
                    {
                        "prefix": bytearray(),
                        "lines": [],
                        "suffix": bytearray(),
                        "length": 0,
                    }
                )
                block = self._read_block_by_threads()
                total_len = block["length"]

            # if the block is 0, it means all the file is processed, check if left epoch is not 0.
            if total_len == 0:
                self.left_epochs -= 1
                # if the left epoch is 0, add a finish indicator to the queue.
                if self.left_epochs == 0:
                    self.outputs.put(FetchDone())
                    self.stop = True
                else:  # otherwise, reset the file index and offset
                    self.outputs.put(
                        {
                            "prefix": bytearray(),
                            "lines": [],
                            "suffix": bytearray(),
                            "length": 0,
                        }
                    )
                    self.file_idx = 0
                    self.offset = self.get_offset()
                    if self.shuffle:
                        random.shuffle(self.files)
            else:
                self.outputs.put(block)
                self.offset += total_len

    def get_worker_files(
        self, path: str, worker_index: int = 0, num_workers: int = 1
    ) -> List[str]:
        # if read files from local path.
        if not hasattr(self, "adl"):
            if self.fs.isfile(path):
                total_files = [path]
            else:
                total_files = [
                    f if isinstance(f, str) else f["name"] for f in self.fs.ls(path)
                ]
        else:
            # read files from azure data lake.
            total_files = self.adl.glob(path)

        if len(total_files) < num_workers:
            raise RuntimeError(
                f"Files:{total_files} are less than number of workers:{num_workers}"
            )
        total_files.sort()
        files = []
        for i in range(worker_index, len(total_files), num_workers):
            files.append(total_files[i])
        return files

    def __iter__(self) -> Iterator[List[str]]:
        self.reset()
        return self

    def __del__(self):
        self.join()

    def _drain_outputs(self):
        while self.outputs.qsize() > 0:
            self.outputs.get()
            self.outputs.task_done()

    def join(self):
        if not threading.main_thread().is_alive():
            self.pool.shutdown(wait=False)
        else:
            self.stop = True
            while not all([f.done() for f in self.workers]):
                # This drain function is used to unblock the threads in
                # thread pool which are blocked by calling "outputs.put"
                # function.
                self._drain_outputs()

            # Sometimes thread in the pool may exit very quickly and the
            # above "while not all([f.done() for f in self.workers])"
            # returns false, we need to call drain function one more time
            # otherwise the following "outputs.join" will block due to
            # un-empty queue.
            self._drain_outputs()

            self.outputs.join()
            self.pool.shutdown(wait=True)

    def __next__(self) -> List[str]:
        # if all the queue item is processed, return stop
        if self.end:
            raise StopIteration

        buf: List[str] = []
        head: dict = {}
        while len(buf) < self.batch_size:
            # try to get lines from existing remain_lines
            if len(self.remain_lines) > 0:
                add_count = self.batch_size - len(buf)
                if add_count > len(self.remain_lines):
                    buf.extend(self.remain_lines)
                    self.remain_lines = []
                else:
                    buf.extend(self.remain_lines[:add_count])
                    self.remain_lines = self.remain_lines[add_count:]
                    break

            # try to get the head of the queue
            head = self.outputs.get()
            self.outputs.task_done()
            # if the head is finish indicator, convert the remaining unprocessed bytes to string and return
            if type(head) is FetchDone:
                if len(self.remain_bytes) != 0:
                    line = self.remain_bytes.decode("utf-8").strip("\r\n")
                    if len(line) > 0:
                        buf.append(line)
                break

            if len(head["prefix"]) > 0:
                self.remain_bytes += head["prefix"]

            if head["suffix"] is not None:
                if len(self.remain_bytes) > 0:
                    self.remain_lines.extend(
                        self.remain_bytes.decode("utf-8").splitlines(False)
                    )
                self.remain_lines.extend(head["lines"])
                self.remain_bytes = head["suffix"]

        # if the batch list count is smaller than batch size, it means the last batch reaches.
        if type(head) is FetchDone or len(buf) != self.batch_size:
            self.end = True

        # if nothing read or last batch less than batch size, stop the iteration
        if len(buf) == 0 or (self.drop_last and len(buf) < self.batch_size):
            raise StopIteration

        return buf

    def __len__(self) -> int:
        raise NotImplementedError


class TextFileSplitIterator(TextFileIterator):
    def __init__(
        self,
        filename: str,
        store_name: str = None,
        adl_config: str = None,
        batch_size: int = 512,
        read_block_in_M: int = 50,
        buffer_queue_size: int = 3,
        thread_count: int = 10,
        worker_offset: int = 0,
        total_read_length: int = -1,
    ):
        self.worker_offset = worker_offset
        self.total_read_length = total_read_length

        super(TextFileSplitIterator, self).__init__(
            filename=filename,
            store_name=store_name,
            adl_config=adl_config,
            batch_size=batch_size,
            epochs=1,
            read_block_in_M=read_block_in_M,
            buffer_queue_size=buffer_queue_size,
            thread_count=thread_count,
            worker_index=0,
            num_workers=1,
            shuffle=False,
            drop_last=False,
        )

    def get_worker_files(
        self, path: str, worker_index: int = 0, num_workers: int = 1
    ) -> List[str]:
        # if read files from local path.
        if not hasattr(self, "adl"):
            total_files = glob.glob(path)
        else:
            # read files from azure data lake.
            total_files = self.adl.glob(path)

        if len(total_files) > 1:
            raise RuntimeError(f"Files:{total_files} are more than 1.")

        return total_files

    def get_length_to_read(self) -> int:
        if self.total_read_length == -1:
            return self.length

        read = min(self.total_read_length, self.length)
        self.total_read_length -= read
        return read

    def get_offset(self) -> int:
        return self.worker_offset
