# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Conversion functions to internal binary format."""
from typing import Optional
import multiprocessing as mp
import math
from operator import add
import fsspec
import json
from deepgnn import get_logger
from deepgnn.graph_engine._adl_reader import TextFileIterator
from deepgnn.graph_engine._base import get_fs
from deepgnn.graph_engine.snark.decoders import DecoderType, JsonDecoder
from deepgnn.graph_engine.snark.converter.writers import BinaryWriter
from deepgnn.graph_engine.snark.dispatcher import (
    PipeDispatcher,
    Dispatcher,
)
from deepgnn.graph_engine.snark.meta import BINARY_DATA_VERSION


class MultiWorkersConverter:
    """Distributed converter implementation."""

    def __init__(
        self,
        graph_path: str,
        output_dir: str,
        decoder: Optional[DecoderType] = None,
        partition_count: int = 1,
        worker_index: int = 0,
        worker_count: int = 1,
        record_per_step: int = 512,
        buffer_size: int = 50,
        queue_size: int = 30,
        thread_count: int = mp.cpu_count(),
        dispatcher: Optional[Dispatcher] = None,
        skip_node_sampler: bool = False,
        skip_edge_sampler: bool = False,
        file_iterator: Optional[TextFileIterator] = None,
        debug: bool = False,
        watermark: Optional[int] = None,
    ):
        """Run multi worker converter in multi process.

        * Converter may stall if an error is thrown inside of the multiprocessing.
        In order to disable multprocessing and see the error message, use debug=True.

        Args:
            graph_path: the raw graph file folder.
            output_dir: the output directory to put the generated graph binary files.
            decoder (Decoder): Decoder object which is used to parse the raw graph data file.
            partition_count: how many partitions will be generated.
            worker_index: If using multiple MultiWorkersConverters, worker_index is the id of this converter.
            worker_count: If using multiple MultiWorkersConverters, worker_count is the number of converters.
            record_per_step: how many lines will be read from the raw graph file to process in each step.
            buffer_size: buffer size in MB used to process the raw graph lines.
            queue_size: the queue size when generating the bin files.
            thread_count: how many thread will be started to process the raw graph in parallel.
            dispatcher: dispatcher type when generating bin files.
            skip_node_sampler(bool): skip generation of node alias tables.
            skip_edge_sampler(bool): skip generation of edge alias tables.
            file_iterator(TextFileIterator): Iterator to yield lines of the input text file.
            debug(bool, False): Enable debug mode to disable multiprocessing and see error messages, forces worker_count=1, paritition_count=1.
            watermark(Optional[int]): for temporal graphs is latest known timestamp in the graph.
        """
        if decoder is None:
            decoder = JsonDecoder()  # type: ignore
        self._debug = debug
        self.graph_path = graph_path
        self.worker_index = worker_index
        self.worker_count = worker_count if not self._debug else 1
        self.output_dir = output_dir
        self.decoder = decoder
        self.record_per_step = record_per_step
        self.read_block_in_M = buffer_size
        self.buffer_queue_size = queue_size
        self.thread_count = thread_count
        self.dispatcher = dispatcher
        self.file_iterator = file_iterator
        self.watermark = watermark

        self.fs, _ = get_fs(graph_path)

        # calculate the partition offset and count of this worker.
        self.partition_count = (
            int(math.ceil(partition_count / worker_count)) if not self._debug else 1
        )
        self.partition_offset = self.partition_count * worker_index
        if self.partition_offset + self.partition_count > partition_count:
            self.partition_count = partition_count - self.partition_offset

        if self.dispatcher is None and not self._debug:
            # when converting data in HDFS make sure to turn it off: https://hdfs3.readthedocs.io/en/latest/limitations.html
            use_threads = True
            if hasattr(fsspec.implementations, "hdfs") and isinstance(
                self.fs, fsspec.implementations.hdfs.PyArrowHDFS
            ):
                use_threads = False
            self.dispatcher = PipeDispatcher(
                self.output_dir,
                self.partition_count,
                self.decoder,  # type: ignore
                partition_offset=self.partition_offset,
                use_threads=use_threads,
                skip_node_sampler=skip_node_sampler,
                skip_edge_sampler=skip_edge_sampler,
                watermark=watermark,
            )

    def convert(self):
        """Convert function."""
        if self.partition_count <= 0:
            return

        get_logger().info(
            f"worker {self.worker_index} try to generate partition: {self.partition_offset} - {self.partition_count + self.partition_offset}"
        )

        if self.file_iterator is None:
            self.file_iterator = TextFileIterator(
                filename=self.graph_path,
                store_name=None,
                batch_size=self.record_per_step,
                epochs=1,
                read_block_in_M=self.read_block_in_M,
                buffer_queue_size=self.buffer_queue_size,
                thread_count=self.thread_count,
                worker_index=self.worker_index,
                num_workers=self.worker_count,
            )

        if not self._debug:
            d = self.dispatcher
            for _, data in enumerate(self.file_iterator):
                for line in data:
                    d.dispatch(line)

            d.join()
            gettr = lambda key: d.prop(key)  # noqa: E731
            partitions = sorted(gettr("partitions"), key=lambda x: x["id"])
            partition_gettr = lambda p, key: p[key]  # noqa: E731
        else:
            assert (
                self.partition_count == 1
            ), "Debug mode does not support multiple partitions."
            if isinstance(self.decoder, type):
                self.decoder = self.decoder()
            writer = BinaryWriter(
                self.output_dir,
                0,
                watermark=self.watermark,
            )
            for _, data in enumerate(self.file_iterator):
                for line in data:
                    print(f"lise is {line}")
                    writer.add(self.decoder.decode(line))
            writer.close()
            gettr = lambda key: getattr(writer, key)  # noqa: E731
            partitions = [writer]
            partition_gettr = lambda p, key: getattr(p, key)  # noqa: E731

        fs, _ = get_fs(self.output_dir)

        mjson = {
            "binary_data_version": BINARY_DATA_VERSION,
            "node_count": gettr("node_count"),
            "edge_count": gettr("edge_count"),
            "node_type_count": gettr("node_type_num"),
            "edge_type_count": gettr("edge_type_num"),
            "node_feature_count": gettr("node_feature_count"),
            "edge_feature_count": gettr("edge_feature_count"),
        }

        edge_count_per_type = [0] * int(gettr("edge_type_num"))
        node_count_per_type = [0] * int(gettr("node_type_num"))
        mjson["partitions"] = {}
        for p in partitions:
            edge_count_per_type = list(
                map(add, edge_count_per_type, partition_gettr(p, "edge_type_count"))
            )
            node_count_per_type = list(
                map(add, node_count_per_type, partition_gettr(p, "node_type_count"))
            )

            i = p["id"] if isinstance(p, dict) else 0

            mjson["partitions"][f"{i}"] = {
                "node_weight": partition_gettr(p, "node_weight"),
                "edge_weight": partition_gettr(p, "edge_weight"),
            }

        mjson["node_count_per_type"] = node_count_per_type
        mjson["edge_count_per_type"] = edge_count_per_type
        mjson["watermark"] = -1 if self.watermark is None else self.watermark

        with fs.open(
            "{}/meta{}.json".format(
                self.output_dir,
                "" if self.worker_count == 1 else f"_{self.worker_index}",
            ),
            "w",
        ) as file:
            file.write(json.dumps(mjson))


if __name__ == "__main__":
    # import here for special usage of the module.
    import argparse
    import deepgnn.graph_engine.snark.decoders as decoders

    parser = argparse.ArgumentParser(
        description="Convert graph from euler json format to the deepgnn binary."
    )
    parser.add_argument(
        "-d", "--data", help="Data file with list of nodes in json format"
    )
    parser.add_argument(
        "-p", "--partitions", type=int, help="Number of partitions to create"
    )
    parser.add_argument(
        "-i", "--worker_index", type=int, default=0, help="Worker index."
    )
    parser.add_argument(
        "-n", "--worker_count", type=int, default=1, help="Number of workers"
    )
    parser.add_argument("-o", "--out", help="Output folder to store binary data")
    parser.add_argument(
        "-t",
        "--type",
        type=str,
        default="json",
        help="Type of decoder object which is used to parse the raw graph data file. Supported: json, tsv",
    )
    parser.add_argument(
        "--skip_node_sampler",
        type=bool,
        default=False,
        help="Skip generation of node alias tables for node sampling",
    )
    parser.add_argument(
        "--skip_edge_sampler",
        type=bool,
        default=False,
        help="Skip generation of edge alias tables for edge sampling",
    )
    parser.add_argument(
        "--watermark",
        type=int,
        default=-1,
        help="Watermark(latest timestamp) to use for temporal graphs. -1 for other graphs",
    )
    args = parser.parse_args()

    decoder = getattr(decoders, f"{args.type.capitalize()}Decoder")()
    if decoder is None:
        raise ValueError("Unsupported decoder type.")

    c = MultiWorkersConverter(
        graph_path=args.data,
        partition_count=args.partitions,
        output_dir=args.out,
        worker_index=args.worker_index,
        worker_count=args.worker_count,
        decoder=decoder,
        skip_node_sampler=args.skip_node_sampler,
        skip_edge_sampler=args.skip_edge_sampler,
        watermark=args.watermark,
    )
    c.convert()
