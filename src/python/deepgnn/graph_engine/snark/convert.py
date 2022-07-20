# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Conversion functions to internal binary format."""
import tempfile
import multiprocessing as mp
import math
from operator import add
import fsspec
from deepgnn import get_logger
from deepgnn.graph_engine._adl_reader import TextFileIterator
from deepgnn.graph_engine._base import get_fs
from deepgnn.graph_engine.snark.decoders import (
    DecoderType,
)
from deepgnn.graph_engine.snark.dispatcher import (
    PipeDispatcher,
    Dispatcher,
)
import deepgnn.graph_engine.snark.meta as mt


class MultiWorkersConverter:
    """Distributed converter implementation."""

    def __init__(
        self,
        graph_path: str,
        meta_path: str,
        output_dir: str,
        decoder_type: DecoderType = DecoderType.JSON,
        partition_count: int = 1,
        worker_index: int = 0,
        worker_count: int = 1,
        record_per_step: int = 512,
        buffer_size: int = 50,
        queue_size: int = 30,
        thread_count: int = mp.cpu_count(),
        dispatcher: Dispatcher = None,
        skip_node_sampler: bool = False,
        skip_edge_sampler: bool = False,
    ):
        """Run multi worker converter in multi process.

        Args:
            graph_path: the raw graph file folder.
            meta_path: the path of the meta.json.
            output_dir: the output directory to put the generated graph binary files.
            decoder_type: decoder type.
            partition_count: how many partitions will be generated.
            worker_index: the work index when running in multi worker mode.
            worker_count: how many workers will be started to convert the data.
            record_per_step: how many lines will be read from the raw graph file to process in each step.
            buffer_size: buffer size in MB used to process the raw graph lines.
            queue_size: the queue size when generating the bin files.
            thread_count: how many thread will be started to process the raw graph in parallel.
            dispatcher: dispatcher type when generating bin files.
            skip_node_sampler(bool): skip generation of node alias tables.
            skip_edge_sampler(bool): skip generation of edge alias tables.
        """
        self.graph_path = graph_path
        self.meta_path = meta_path
        self.worker_index = worker_index
        self.worker_count = worker_count
        self.output_dir = output_dir
        self.decoder_type = decoder_type
        self.record_per_step = record_per_step
        self.read_block_in_M = buffer_size
        self.buffer_queue_size = queue_size
        self.thread_count = thread_count
        self.dispatcher = dispatcher

        self.fs, _ = get_fs(graph_path)
        # download the meta.json to local folder for dispatcher.
        tmp_folder = tempfile.TemporaryDirectory()
        meta_path_local = mt._get_meta_path(tmp_folder.name)
        self.fs.get_file(meta_path, meta_path_local)

        # calculate the partition offset and count of this worker.
        self.partition_count = int(math.ceil(partition_count / worker_count))
        self.partition_offset = self.partition_count * worker_index
        if self.partition_offset + self.partition_count > partition_count:
            self.partition_count = partition_count - self.partition_offset

        if self.dispatcher is None:
            # when converting data in HDFS make sure to turn it off: https://hdfs3.readthedocs.io/en/latest/limitations.html
            use_threads = True
            if hasattr(fsspec.implementations, "hdfs") and isinstance(
                self.fs, fsspec.implementations.hdfs.PyArrowHDFS
            ):
                use_threads = False
            self.dispatcher = PipeDispatcher(
                self.output_dir,
                self.partition_count,
                meta_path_local,
                self.decoder_type,
                partition_offset=self.partition_offset,
                use_threads=use_threads,
                skip_node_sampler=skip_node_sampler,
                skip_edge_sampler=skip_edge_sampler,
            )

    def convert(self):
        """Convert function."""
        if self.partition_count <= 0:
            return

        get_logger().info(
            f"worker {self.worker_index} try to generate partition: {self.partition_offset} - {self.partition_count + self.partition_offset}"
        )

        d = self.dispatcher
        dataset = TextFileIterator(
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

        for _, data in enumerate(dataset):
            for line in data:
                d.dispatch(line)

        d.join()

        fs, _ = get_fs(self.output_dir)
        with fs.open(
            "{}/meta{}.txt".format(
                self.output_dir,
                "" if self.worker_count == 1 else f"_{self.worker_index}",
            ),
            "w",
        ) as mtxt:
            mtxt.writelines(
                [
                    str(d.prop("node_count")),
                    "\n",
                    str(d.prop("edge_count")),
                    "\n",
                    str(d.prop("node_type_num")),
                    "\n",
                    str(d.prop("edge_type_num")),
                    "\n",
                    str(
                        int(d.prop("node_float_feature_num"))
                        + int(d.prop("node_binary_feature_num"))
                        + int(d.prop("node_uint64_feature_num"))
                    ),
                    "\n",
                    str(
                        int(d.prop("edge_float_feature_num"))
                        + int(d.prop("edge_binary_feature_num"))
                        + int(d.prop("edge_uint64_feature_num"))
                    ),
                    "\n",
                    str(len(d.prop("partitions"))),
                    "\n",
                ]
            )

            edge_count_per_type = [0] * int(d.prop("edge_type_num"))
            node_count_per_type = [0] * int(d.prop("node_type_num"))
            for p in sorted(d.prop("partitions"), key=lambda x: x["id"]):
                edge_count_per_type = list(
                    map(add, edge_count_per_type, p["edge_type_count"])
                )
                node_count_per_type = list(
                    map(add, node_count_per_type, p["node_type_count"])
                )

                mtxt.writelines([str(p["id"]), "\n"])
                for nw in p["node_weight"]:
                    mtxt.writelines([str(nw), "\n"])

                for ew in p["edge_weight"]:
                    mtxt.writelines([str(ew), "\n"])
            for count in node_count_per_type:
                mtxt.writelines([str(count), "\n"])
            for count in edge_count_per_type:
                mtxt.writelines([str(count), "\n"])


if __name__ == "__main__":
    # import here for special usage of the module.
    import argparse

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
    parser.add_argument(
        "-m", "--meta", help="Metadata about graph: number of node, types, etc"
    )
    parser.add_argument("-o", "--out", help="Output folder to store binary data")
    parser.add_argument(
        "-t",
        "--type",
        type=DecoderType,
        default=DecoderType.JSON,
        help="Type of the graph data file. Supported: json, tsv",
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
    args = parser.parse_args()

    c = MultiWorkersConverter(
        graph_path=args.data,
        meta_path=args.meta,
        partition_count=args.partitions,
        output_dir=args.out,
        worker_index=args.worker_index,
        worker_count=args.worker_count,
        decoder_type=args.type,
        skip_node_sampler=args.skip_node_sampler,
        skip_edge_sampler=args.skip_edge_sampler,
    )
    c.convert()
