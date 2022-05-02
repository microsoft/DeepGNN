# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Orchestrate multiple converters."""
import json
import multiprocessing as mp
import threading
import platform
import typing
from abc import ABC, abstractmethod

if platform.system() == "Windows":
    from multiprocessing.connection import PipeConnection as Connection  # type: ignore
else:
    from multiprocessing.connection import Connection  # type: ignore

from deepgnn.graph_engine.snark.decoders import DecoderType
from deepgnn import get_logger

FLAG_ALL_DONE = b"WORK_FINISHED"
FLAG_WORKER_FINISHED_PROCESSING = b"WORKER_FINISHED_PROCESSING"
PROCESS_PRINT_INTERVAL = 1000


class Dispatcher(ABC):
    """Interface for dispatching graph elements across binary writers."""

    @abstractmethod
    def dispatch(self, line: str):
        """Dispath a line representing a graph element.

        Args:
            line (str): graph element represented as string.
        """
        pass

    @abstractmethod
    def join(self):
        """Wait to finish work."""
        pass

    @abstractmethod
    def prop(self, name: str) -> typing.Any:
        """Properties relevant to convertion(partition count, etc) or graph metadata.

        Args:
            name (str): property name.
        """
        pass


class PipeDispatcher(Dispatcher):
    """Use multiprocessing.Pipe to distribute node json objects to workers."""

    def __init__(
        self,
        folder: str,
        parallel: int,
        meta: str,
        process: typing.Callable[
            [
                typing.Union[mp.Queue, Connection],
                mp.Queue,
                str,
                int,
                int,
                int,
                DecoderType,
                bool,
                bool,
            ],
            None,
        ],
        decoder_type: DecoderType,
        partition_offset: int = 0,
        use_threads: bool = False,
        skip_node_sampler: bool = False,
        skip_edge_sampler: bool = False,
    ):
        """Create dispatcher.

        Args:
            folder (str): Location of graph files.
            parallel (int): Number of parallel process to use for conversion.
            meta (str): Meta data about graph.
            process (typing.Callable[ [typing.Union[mp.Queue, Connection], mp.Queue, str, int, int, int, DecoderType], None ]): Function to call for processing lines in a file.
            decoder_type (DecoderType): Decoder which is used to parse the raw graph data file, Supported: json/tsv.
            partition_offset(int): offset in a text file, where to start reading for a new partition.
            use_threads(bool): use threads instead of processes for parallel processing.
            skip_node_sampler(bool): skip generation of node alias tables.
            skip_edge_sampler(bool): skip generation of edge alias tables.
        """
        super().__init__()
        with open(meta, "r") as fm:
            self.jsm = json.load(fm)

        parallel_func = mp.Process  # type: ignore
        if use_threads:
            parallel_func = threading.Thread  # type: ignore

        self.parallel = parallel
        self.count = 0
        self.q_in = [mp.Pipe(False) for _ in range(parallel)]
        self.q_out: mp.Queue = mp.Queue(parallel)
        processes = [
            parallel_func(
                target=process,
                args=(
                    self.q_in[i][0],
                    self.q_out,
                    folder,
                    i + partition_offset,
                    int(self.jsm["node_type_num"]),
                    int(self.jsm["edge_type_num"]),
                    decoder_type,
                    skip_node_sampler,
                    skip_edge_sampler,
                ),
            )
            for i in range(parallel)
        ]
        for p in processes:
            p.start()

        self.node_count = 0
        self.edge_count = 0
        self.partitions: typing.List[typing.Dict] = []

    def dispatch(self, line: str):
        """Dispatch a line to a process.

        Args:
            line (str): graph element.
        """
        self.q_in[self.count % self.parallel][1].send(line)
        self.count += 1

        if self.count % PROCESS_PRINT_INTERVAL == 0:
            get_logger().info(f"record processed: {self.count}")

    def join(self):
        """Wait for all processes to finish work."""
        for i in range(self.parallel):
            self.q_in[i][1].send(FLAG_ALL_DONE)

        for _ in range(self.parallel):
            flag, output = self.q_out.get()
            self.node_count += output["node_count"]
            self.edge_count += output["edge_count"]
            self.partitions.append(output["partition"])

            assert flag == FLAG_WORKER_FINISHED_PROCESSING
        self.q_out.close()

    def prop(self, name: str) -> typing.Any:
        """Properties relevant for conversion.

        Args:
            name (str): property name.

        Returns:
            typing.Any: property value.
        """
        name = name.lower()
        if name == "node_count":
            return self.node_count
        if name == "edge_count":
            return self.edge_count
        if name == "partitions":
            return self.partitions

        return self.jsm[name]


class QueueDispatcher(Dispatcher):
    """
    QueueDispatcher is using queues to distribute work among multiple processes.

    It is a more flexible implementation of a dispatcher:
    * input queues are limited in size
    * partition_func may be used to distribute nodes to desired partitions
    """

    def __init__(
        self,
        folder: str,
        num_partitions: int,
        meta: str,
        process: typing.Callable[[mp.Queue, mp.Queue, str, int, int], None],
        partion_func: typing.Callable[[str], int],
        decoder_type: DecoderType,
        partition_offset: int = 0,
        use_threads: bool = False,
        skip_node_sampler: bool = False,
        skip_edge_sampler: bool = False,
    ):
        """Create dispatcher based on the queue.

        Args:
            folder (str): Location of graph files.
            num_partitions (int): number of binary partitions to create.
            meta (str): meta data about graph.
            process (typing.Callable[[mp.Queue, mp.Queue, str, int, int], None]): function to use for conversion.
            partion_func (typing.Callable[[str], int]): how to assign graph elements to a partition.
            decoder_type (DecoderType): Decoder which is used to parse the raw graph data file, Supported: json/tsv.
            partition_offset(int): offset in a text file, where to start reading for a new partition.
            use_threads(bool): use threads instead of processes for parallel processing.
            skip_node_sampler(bool): skip generation of node alias tables.
            skip_edge_sampler(bool): skip generation of edge alias tables.
        """
        super().__init__()
        with open(meta, "r") as fm:
            self.jsm = json.load(fm)

        parallel_func = mp.Process  # type: ignore
        if use_threads:
            parallel_func = threading.Thread  # type: ignore

        self.count = 0
        self.num_partitions = num_partitions
        self.q_in: typing.List[mp.Queue] = [mp.Queue(2) for _ in range(num_partitions)]
        self.q_out: mp.Queue = mp.Queue(num_partitions)
        self.partition_func = partion_func

        # not all partitions might be used, we'll keep track them here
        self.active_partitions: typing.Set = set()
        self.processes = [
            parallel_func(
                target=process,
                args=(
                    self.q_in[i],
                    self.q_out,
                    folder,
                    i + partition_offset,
                    int(self.jsm["node_type_num"]),
                    int(self.jsm["edge_type_num"]),
                    decoder_type,
                    skip_node_sampler,
                    skip_edge_sampler,
                ),
            )
            for i in range(num_partitions)
        ]
        for p in self.processes:
            p.daemon = True
            p.start()

        self.node_count = 0
        self.edge_count = 0
        self.partitions: typing.List[typing.Dict] = []

    def dispatch(self, line: str):
        """Dispath a line to one of the queues.

        Args:
            line (str): String representation of a graph element.
        """
        partition = self.partition_func(line)
        self.q_in[partition].put(line)
        self.active_partitions.add(partition)

        self.count += 1
        if self.count % PROCESS_PRINT_INTERVAL == 0:
            get_logger().info(f"record processed: {self.count}")

    def join(self):
        """Wrapup conversion."""
        for i in self.active_partitions:
            self.q_in[i].put(FLAG_ALL_DONE)
            self.q_in[i].close()

        for _ in self.active_partitions:
            flag, output = self.q_out.get()
            self.node_count += output["node_count"]
            self.edge_count += output["edge_count"]
            self.partitions.append(output["partition"])

            assert flag == FLAG_WORKER_FINISHED_PROCESSING
        self.q_out.close()

        for p in self.processes:
            p.terminate()

    def prop(self, name: str) -> typing.Any:
        """Properties relevant for conversion.

        Args:
            name (str): property name.

        Returns:
            typing.Any: property value.
        """
        name = name.lower()
        if name == "node_count":
            return self.node_count
        if name == "edge_count":
            return self.edge_count
        if name == "partitions":
            return self.partitions

        return self.jsm[name]
