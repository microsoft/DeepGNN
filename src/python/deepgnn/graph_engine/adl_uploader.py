# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Classes to upload data to ADL."""
import threading
import queue
import multiprocessing
from multiprocessing import Queue
import time
from azure.datalake.store import core, lib
from deepgnn import get_logger
from deepgnn.graph_engine._adl_reader import AdlCredentialParser


class MultiThreadsAdlDataWriter:
    """
    Uploading processor which is created by AdlUploader.

    It has its own ADL file system and create multiple threads to upload data.
    """

    def __init__(
        self,
        adl_config,
        process_idx: int,
        store_name: str,
        file_path_prefix: str,
        max_chunks_per_file: int = 0,  # 0 : no limitation for file size.
        threads_per_process: int = 5,
    ):
        """Initialize writer."""
        self.logger = get_logger()
        self.process_idx = process_idx
        self.data_queue: queue.Queue = queue.Queue()
        self.file_path_prefix = file_path_prefix
        self.max_chunks_per_file = max_chunks_per_file
        principal_token = lib.auth(
            tenant_id=adl_config["TENANT_ID"],
            client_secret=adl_config["CLIENT_SECRET"],
            client_id=adl_config["CLIENT_ID"],
        )
        self.adl = core.AzureDLFileSystem(token=principal_token, store_name=store_name)
        self.threads = []
        for i in range(threads_per_process):
            thread = threading.Thread(target=self._create_worker, args=(i,))
            thread.name = str(i)
            self.threads.append(thread)
        self.daemon_thread = threading.Thread(target=self._monitor_thread)
        self.started = False
        self.start()

    def _monitor_thread(self):
        while True:
            all_alive = True
            for thread in self.threads:
                if not thread.is_alive():
                    self.logger.warn(
                        f"Thread {thread.name} in process {self.process_idx} exited early!"
                    )
                    all_alive = False
                    break
            if not all_alive:
                self.logger.warn(
                    f"Process {self.process_idx} will exit due to some thread stop early."
                )
                self.stop()
                break
            time.sleep(10)

    def _create_worker(self, thread_idx: int):
        self.logger.info(
            f"[AdlUploader] start thread {thread_idx} for processor {self.process_idx}."
        )

        chunk_count = 0
        file_index = 0
        out_file = None
        while True:
            chunk = self.data_queue.get()
            if not chunk:
                if out_file:
                    out_file.close()
                self.data_queue.task_done()

                self.logger.info(f"thread {thread_idx} exit.")

                break
            if chunk_count == 0 or (
                self.max_chunks_per_file > 0
                and chunk_count % self.max_chunks_per_file == 0
            ):
                if out_file:
                    out_file.close()
                filename = f"_{self.process_idx}_{thread_idx}_{file_index}.tsv"
                file_path = self.file_path_prefix + filename
                out_file = self.adl.open(file_path, mode="wb")
                file_index += 1
            if out_file:
                out_file.write(chunk)
            self.data_queue.task_done()
            chunk_count += 1

    def enqueue(self, chunk):
        """Put data in processing queue."""
        self.data_queue.put(chunk)

    def start(self):
        """Start upload."""
        if not self.started:
            self.logger.info(f"[uploader processor-{self.process_idx}] start...")

            for thread in self.threads:
                thread.start()

            self.daemon_thread.daemon = True
            self.daemon_thread.start()

            self.started = True
            self.logger.info(
                f"[uploader processor-{self.process_idx}] started successfully."
            )

    def stop(self):
        """Stop uploading data."""
        if self.started:
            self.logger.info(f"[uploader processor-{self.process_idx}] stop...")
            self.started = False

            for thread in self.threads:
                if thread.is_alive():
                    self.data_queue.put(None)

            for thread in self.threads:
                thread.join()

            self.logger.info(
                f"[uploader processor-{self.process_idx}] stopped successfully."
            )

    def is_alive(self):
        """Check writer status."""
        return self.started


class AdlDataWriter:
    """
    Leverage azure datalake SDK and upload data to adl.

    To achieve high throughput, multi-process mode is needed, and each process creates multiple threads to upload data.
    """

    def __init__(
        self,
        store_name: str,
        file_path_prefix: str,
        max_chunks_per_file: int = 0,
        process_num: int = 5,
        threads_per_process: int = 5,
        queue_size: int = 100,
        max_lines_per_chunk: int = 1024,
    ):
        """Initialize writer."""
        self.logger = get_logger()
        self.max_lines_per_chunk = max_lines_per_chunk
        self.max_chunks_per_file = max_chunks_per_file
        self.threads_per_process = threads_per_process
        self.store_name = store_name
        self.file_path_prefix = file_path_prefix
        self.queue_size = queue_size
        self.processes = []
        self.adl_config = AdlCredentialParser.read_credentials()
        self.data_queues = []

        for i in range(process_num):
            data_queue: Queue = multiprocessing.Queue(int(queue_size / process_num))
            data_queue.cancel_join_thread()
            process = multiprocessing.Process(target=self._start_process, args=(i,))
            process.name = str(i)
            process.daemon = True
            self.processes.append(process)
            self.data_queues.append(data_queue)

        self.daemon_thread = threading.Thread(target=self._monitor_thread)

        self.total_chunk = 0
        self.is_running = False
        self._start_uploading()

    def _monitor_thread(self):
        while True:
            all_alive = True
            for process in self.processes:
                if not process.is_alive():
                    self.logger.warn(f"Process {process.name} exited early!")
                    all_alive = False
                    break
            if not all_alive:
                self.logger.warn(
                    "All processes will exit due to some of them stop early."
                )
                self.close()
                break
            time.sleep(30)

    def _start_process(self, process_idx: int):
        self.logger.info(f"[AdlUploader] start process {process_idx}.")
        uploader = MultiThreadsAdlDataWriter(
            process_idx=process_idx,
            adl_config=self.adl_config,
            max_chunks_per_file=self.max_chunks_per_file,
            threads_per_process=self.threads_per_process,
            store_name=self.store_name,
            file_path_prefix=self.file_path_prefix,
        )
        while True:
            chunk = self.data_queues[process_idx].get()
            if not chunk or not uploader.is_alive():
                break
            uploader.enqueue(chunk)

        self.logger.info(f"[AdlUploader] stop process {process_idx}.")
        uploader.stop()

    def write(self, embed):
        """Write data to processing queue."""
        chunk = bytes(embed, "utf-8")
        self.data_queues[self.total_chunk % len(self.processes)].put(chunk)
        self.total_chunk += 1

    def writelines(self, embeds):
        """Write data line by line."""
        offset = 0
        embed_size = len(embeds)
        embeds = [str(x) for x in embeds]
        while offset < embed_size:
            end_offset = (
                offset + self.max_lines_per_chunk
                if offset + self.max_lines_per_chunk < embed_size
                else embed_size
            )
            if not self.is_running:
                return False
            chunk = "\n".join(embeds[offset:end_offset]) + "\n"
            chunk = bytes(chunk, "utf-8")
            try:
                self.data_queues[self.total_chunk % len(self.processes)].put(
                    chunk, timeout=1
                )
            except ValueError:
                self.total_chunk += 1

            offset = end_offset
            self.total_chunk += 1
        return True

    def _start_uploading(self):
        for process in self.processes:
            process.start()
        self.daemon_thread.daemon = True
        self.daemon_thread.start()

        self.is_running = True
        self.logger.info("[AdlUploader] started successfully.")

    def close(self):
        """Stop uploading and put close flags to data queues."""
        if self.is_running:
            self.logger.info("[AdlUploader] stop...")
            self.is_running = False

            for i in range(len(self.processes)):
                if self.processes[i].is_alive():
                    self.data_queues[i].put(None)

            for process in self.processes:
                process.join()

            self.logger.info("[AdlUploader] stopped successfully.")

    def __enter__(self):
        """Return writer."""
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Finalize uploading."""
        self.close()
