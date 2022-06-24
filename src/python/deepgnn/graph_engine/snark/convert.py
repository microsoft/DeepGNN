# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Conversion functions to internal binary format."""
import tempfile
import multiprocessing as mp
import typing
import math
import platform
from operator import add

if platform.system() == "Windows":
    from multiprocessing.connection import PipeConnection as Connection  # type: ignore
else:
    from multiprocessing.connection import Connection  # type: ignore

import fsspec

from deepgnn import get_logger
from deepgnn.graph_engine._adl_reader import TextFileIterator
from deepgnn.graph_engine._base import get_fs
import deepgnn.graph_engine.snark.converter.converter as converter
import deepgnn.graph_engine.snark.decoders as decoders
from deepgnn.graph_engine.snark.decoders import (
    LinearDecoder,
)
from deepgnn.graph_engine.snark.dispatcher import (
    PipeDispatcher,
    Dispatcher,
    FLAG_ALL_DONE,
    FLAG_WORKER_FINISHED_PROCESSING,
)
import deepgnn.graph_engine.snark.meta as mt


class _NoOpWriter:
    def add(self, *_: typing.Any):
        return

    def add_type(self, _: int):
        return

    def close(self):
        return


def output(
    q_in: typing.Union[mp.Queue, Connection],
    q_out: mp.Queue,
    folder: str,
    suffix: int,
    decoder_class: typing.Any,
    skip_node_sampler: bool,
    skip_edge_sampler: bool,
    meta_data: dict,
) -> None:
    """Process graph nodes from a queue to binary files.

    Args:
        q_in (typing.Union[mp.Queue, Connection]): input nodes
        q_out (mp.Queue): signal processing is done
        folder (str): where to save binaries
        suffix (int): file suffix in the name of binary files
        decoder_class (Decoder): Class of decoder which is used to parse the raw graph data file.
        skip_node_sampler(bool): skip generation of node alias tables
        skip_edge_sampler(bool): skip generation of edge alias tables
    """
    assert decoder_class is not None
    decoder = decoder_class() if isinstance(decoder_class, type) else decoder_class
    decoder.set_metadata(meta_data)
    node_count = 0
    edge_count = 0
    node_weight = []
    node_type_count = []
    edge_weight = []
    edge_type_count = []
    node_writer = converter.NodeWriter(str(folder), suffix)
    edge_writer = converter.EdgeWriter(str(folder), suffix)
    node_alias: typing.Union[converter.NodeAliasWriter, _NoOpWriter] = (
        _NoOpWriter()
        if skip_node_sampler
        else converter.NodeAliasWriter(str(folder), suffix)
    )
    edge_alias: typing.Union[converter.EdgeAliasWriter, _NoOpWriter] = (
        _NoOpWriter()
        if skip_edge_sampler
        else converter.EdgeAliasWriter(str(folder), suffix)
    )
    count = 0
    node_type_num = 0
    edge_type_num = 0
    while True:
        count += 1
        if type(q_in) == Connection:
            lines = q_in.recv()  # type: ignore
        else:
            lines = q_in.get()  # type: ignore

        if lines == FLAG_ALL_DONE:
            break

        for src, dst, typ, weight, features in decoder.decode(lines):
            if src == -1:
                node_writer.add(dst, typ, features)
                edge_writer.add_node()
                if typ + 1 > node_type_num:
                    for i in range(node_type_num, typ + 1):
                        node_type_num += 1
                        node_weight.append(0)
                        node_type_count.append(0)
                        node_alias.add_type(i)
                node_alias.add(dst, typ, weight)
                node_weight[typ] += float(weight)
                node_type_count[typ] += 1
                node_count += 1
            else:
                edge_writer.add(dst, typ, weight, features)
                if typ + 1 > edge_type_num:
                    for i in range(edge_type_num, typ + 1):
                        edge_type_num += 1
                        edge_weight.append(0)
                        edge_type_count.append(0)
                        edge_alias.add_type(i)
                edge_alias.add(src, dst, typ, weight)
                edge_weight[typ] += weight
                edge_type_count[typ] += 1
                edge_count += 1

    node_writer.close()
    edge_writer.close()
    node_alias.close()
    edge_alias.close()
    q_out.put(
        (
            FLAG_WORKER_FINISHED_PROCESSING,
            {
                "node_count": node_count,
                "edge_count": edge_count,
                "partition": {
                    "id": suffix,
                    "node_weight": node_weight,
                    "node_type_count": node_type_count,
                    "edge_weight": edge_weight,
                    "edge_type_count": edge_type_count,
                    "node_type_num": node_type_num,
                    "edge_type_num": edge_type_num,
                    "node_feature_num": node_writer.feature_writer.node_feature_num,
                    "edge_feature_num": edge_writer.feature_writer.edge_feature_num,
                },
            },
        )
    )


class MultiWorkersConverter:
    """Distributed converter implementation."""

    def __init__(
        self,
        graph_path: str,
        output_dir: str,
        meta_path: str = "",
        decoder_class: typing.Any = LinearDecoder,
        partition_count: int = 1,
        worker_index: int = 0,
        worker_count: int = 1,
        record_per_step: int = 512,
        buffer_size: int = 10,
        queue_size: int = 30,
        thread_count: int = mp.cpu_count(),
        dispatcher: Dispatcher = None,
        skip_node_sampler: bool = False,
        skip_edge_sampler: bool = False,
    ):
        """Run multi worker converter in multi process.

        Args:
            graph_path: the raw graph file folder.
            output_dir: the output directory to put the generated graph binary files.
            meta_path(optional): the path of the meta.json.
            decoder_class: decoder type.
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
        self.decoder_class = decoder_class
        self.record_per_step = record_per_step
        self.read_block_in_M = buffer_size
        self.buffer_queue_size = queue_size
        self.thread_count = thread_count
        self.dispatcher = dispatcher

        self.fs, _ = get_fs(graph_path)
        # download the meta.json to local folder for dispatcher.
        if meta_path:
            tmp_folder = tempfile.TemporaryDirectory()
            meta_path_local = mt._get_meta_path(tmp_folder.name)
            self.fs.get_file(meta_path, meta_path_local)
        else:
            meta_path_local = ""

        # calculate the partition offset and count of this worker.
        self.partition_count = int(math.ceil(partition_count / worker_count))
        self.partition_offset = self.partition_count * worker_index
        if self.partition_offset + self.partition_count > partition_count:
            self.partition_count = partition_count - self.partition_offset

        if self.dispatcher is None:
            self.dispatcher = PipeDispatcher(
                self.output_dir,
                self.partition_count,
                output,
                decoder_class,
                meta_path_local,
                self.partition_offset,
                False
                if hasattr(fsspec.implementations, "hdfs")
                and isinstance(self.fs, fsspec.implementations.hdfs.PyArrowHDFS)
                else True,  # when converting data in HDFS make sure to turn it off: https://hdfs3.readthedocs.io/en/latest/limitations.html
                skip_node_sampler,
                skip_edge_sampler,
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

        for data in dataset:
            for line in data:
                d.dispatch(line)

        d.join()

        fs, _ = get_fs(self.output_dir)
        partitions = sorted(d.prop("partitions"), key=lambda x: x["id"])
        node_type_num = max([p["node_type_num"] for p in partitions])
        edge_type_num = max([p["edge_type_num"] for p in partitions])
        node_feature_num = max([p["node_feature_num"] for p in partitions])
        edge_feature_num = max([p["edge_feature_num"] for p in partitions])
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
                    str(node_type_num),
                    "\n",
                    str(edge_type_num),
                    "\n",
                    str(node_feature_num),
                    "\n",
                    str(edge_feature_num),
                    "\n",
                    str(len(partitions)),
                    "\n",
                ]
            )

            node_count_per_type = [0] * int(node_type_num)
            edge_count_per_type = [0] * int(edge_type_num)
            for p in partitions:
                for i, v in enumerate(p["node_type_count"]):
                    node_count_per_type[i] += v
                for i, v in enumerate(p["edge_type_count"]):
                    edge_count_per_type[i] += v

                mtxt.writelines([str(p["id"]), "\n"])
                for i, nw in enumerate(p["node_weight"]):
                    mtxt.writelines([str(nw), "\n"])

                c = converter.NodeAliasWriter(self.output_dir, p["id"])
                for ii in range(i + 1, node_type_num):
                    c.add_type(ii)
                    mtxt.writelines([str(0), "\n"])
                c.close()

                for i, ew in enumerate(p["edge_weight"]):
                    mtxt.writelines([str(ew), "\n"])
                c = converter.EdgeAliasWriter(self.output_dir, p["id"])
                for ii in range(i + 1, edge_type_num):
                    c.add_type(ii)
                    mtxt.writelines([str(0), "\n"])
                c.close()
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
        "-m",
        "--meta",
        default="",
        help="Metadata about graph: number of node, types, etc",
    )
    parser.add_argument("-o", "--out", help="Output folder to store binary data")
    parser.add_argument(
        "-t",
        "--type",
        type=str,
        default="linear",
        help="Type of the graph data file. Supported: linear, json, tsv",
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

    decoder_class = getattr(decoders, f"{args.type.capitalize()}Decoder")
    if decoder_class is None:
        raise ValueError("Unsupported decoder type.")

    c = MultiWorkersConverter(
        graph_path=args.data,
        meta_path=args.meta,
        partition_count=args.partitions,
        output_dir=args.out,
        worker_index=args.worker_index,
        worker_count=args.worker_count,
        decoder_class=decoder_class,
        skip_node_sampler=args.skip_node_sampler,
        skip_edge_sampler=args.skip_edge_sampler,
    )
    c.convert()
