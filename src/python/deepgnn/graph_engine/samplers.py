# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Dataset classes useful for preparing training/evaluation examples."""
import glob
import math
import random

from typing import Callable, List, Tuple, Union
import csv
from deepgnn import get_logger
import numpy as np
from deepgnn.graph_engine._base import SamplingStrategy, Graph
from deepgnn.graph_engine._adl_reader import TextFileIterator

INVALID_NODE_ID = -1


def get_files(filenames: str, worker_index: int, num_workers: int) -> List[str]:
    """Parition sampler files across workers."""
    files = sorted(glob.glob(filenames))
    if len(files) < num_workers:
        raise RuntimeError(
            "input sample #files is less than #workers({0}). files:{1}".format(
                num_workers, ",".join(files)
            )
        )
    sample_files = files[worker_index : len(files) : num_workers]
    return sample_files


def get_feature_type(feature_type: np.dtype) -> type:
    """Map numpy to python types."""
    if feature_type == np.float32:
        return float
    elif feature_type == np.int64:
        return int
    raise RuntimeError("unknown feature_type: {}".format(str(feature_type)))


class _GEIterator:
    """Iterable component to iterate using graph API.

    For example, if the func is graph.sample_nodes,
    each next() function will return a list of nodes queried
    from graph engine servers.
    """

    def __init__(
        self,
        batch_size: int,
        item_type: np.ndarray,
        count: int,
        strategy: SamplingStrategy,
        func: Callable[..., np.ndarray],
    ):
        self.batch_size = batch_size
        self.item_type = item_type
        self.count = count
        self.func = func
        self.strategy = strategy

    def __next__(self):
        self.count -= 1
        if self.count < 0:
            raise StopIteration

        return self.func(self.batch_size, self.item_type, strategy=self.strategy)


class BaseSampler(object):
    """Sampler class for Node/Edge Sampling."""

    def __init__(
        self,
        batch_size: int,
        epochs: int,
        shuffle: bool,
        sample_num: int = -1,
        prefetch_size: int = 4,
        data_parallel_num: int = 1,
        data_parallel_index: int = 0,
    ):
        """Initialize sampler."""
        self.batch_size = batch_size
        self.sample_num = sample_num
        self.epochs = epochs
        self.shuffle = shuffle
        self.prefetch_size = prefetch_size
        self._data_parallel_num = data_parallel_num
        self._data_parallel_index = data_parallel_index
        self.logger = get_logger()

    def __iter__(self) -> _GEIterator:
        """Must be implemented in derived classes."""
        raise NotImplementedError()

    def __count__(self) -> int:
        """Must be implemented in derived classes."""
        raise NotImplementedError()

    def __len__(self) -> int:
        """Must be implemented in derived classes."""
        raise NotImplementedError()

    def reset(self):
        """Skip by default."""
        pass

    @property
    def data_parallel_index(self):
        """Worker index."""
        return self._data_parallel_index

    @data_parallel_index.setter
    def data_parallel_index(self, value):
        """Reset worker index."""
        self._data_parallel_index = value

    @property
    def data_parallel_num(self):
        """Return number of workers."""
        return self._data_parallel_num

    @data_parallel_num.setter
    def data_parallel_num(self, value):
        """Reset number of workers."""
        self._data_parallel_num = value


class GENodeSampler(BaseSampler):
    """Sampler to query node ids from graph engine."""

    def __init__(
        self,
        graph: Graph,
        node_types: Union[int, List[int]],
        batch_size: int,
        epochs: int,
        sample_num: int = -1,
        num_workers: int = 1,
        strategy: SamplingStrategy = SamplingStrategy.Weighted,
        data_parallel_num: int = 1,
        data_parallel_index: int = 0,
    ):
        """Initialize node sampler."""
        super().__init__(
            batch_size,
            epochs,
            shuffle=True,
            sample_num=sample_num,
            data_parallel_index=data_parallel_index,
            data_parallel_num=data_parallel_num,
        )
        self.strategy = strategy
        self.node_types = node_types
        self.graph = graph
        if self.sample_num == -1:
            self.sample_num = graph.node_count(self.node_types)
            self.logger.info(f"Node Count ({self.node_types}): {self.sample_num}")
        self.count = math.ceil(
            (self.epochs * self.sample_num)
            / (self.batch_size * num_workers * data_parallel_num)
        )

    # sample nodes returns node_ids and types if passed node_types as np.array
    def _strip_types(self, *args, **kwargs):
        if isinstance(self.node_types, np.ndarray):
            return self.graph.sample_nodes(*args, **kwargs)[0]
        # return original output if node_types is an integer
        return self.graph.sample_nodes(*args, **kwargs)

    def __iter__(self) -> _GEIterator:
        """Create a graph engine iterator."""
        return _GEIterator(
            self.batch_size,
            self.node_types,
            self.count,
            self.strategy,
            self._strip_types,
        )

    def __len__(self) -> int:
        """Total number of minibatches in this sampler."""
        return self.count


class GEEdgeSampler(BaseSampler):
    """Sampler to query edge ids from graph engine."""

    def __init__(
        self,
        graph: Graph,
        edge_types: Union[int, List[int]],
        batch_size: int,
        epochs: int,
        sample_num: int = -1,
        num_workers: int = 1,
        strategy: SamplingStrategy = SamplingStrategy.Weighted,
        data_parallel_num: int = 1,
        data_parallel_index: int = 0,
    ):
        """Initialize edge sampler."""
        super().__init__(
            batch_size,
            epochs,
            shuffle=True,
            sample_num=sample_num,
            data_parallel_index=data_parallel_index,
            data_parallel_num=data_parallel_num,
        )
        self.strategy = strategy
        self.edge_types = edge_types
        self.graph = graph

        if self.sample_num == -1:
            self.sample_num = graph.edge_count(self.edge_types)
            self.logger.info(f"Edge Count ({self.edge_types}): {self.sample_num}")
        self.count = math.ceil(
            (self.epochs * self.sample_num)
            / (self.batch_size * num_workers * data_parallel_num)
        )

    def __iter__(self) -> _GEIterator:
        """Create a graph engine iterator."""
        return _GEIterator(
            self.batch_size,
            self.edge_types,
            self.count,
            self.strategy,
            self.graph.sample_edges,
        )

    def __len__(self) -> int:
        """Total number of minibatches in this sampler."""
        return self.count


class _NumpyIterator(_GEIterator):
    """Private iterable component to iterate numpy array."""

    def __init__(
        self,
        data: List[np.ndarray],
        batch_size: int,
        epochs: int = 1,
        shuffle: bool = False,
        backfill: List[int] = [],
        drop_last: bool = False,
        data_parallel_num: int = 1,
        data_parallel_index: int = 0,
    ):
        """Initialize iterator."""
        self.length = len(data[0])
        assert self.length > 0
        if len(data) > 1:
            for item in data:
                assert len(item) == self.length
        self.epochs = epochs
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.backfill = backfill
        self.drop_last = drop_last
        self.idx = self.__build_index(epochs, self.length)
        self.data_parallel_num = data_parallel_num
        self.offset = data_parallel_index * self.batch_size

    def __build_index(self, epochs: int, length: int) -> np.ndarray:
        idx = np.empty(epochs * length, np.int64)
        for i in range(epochs):
            t = np.random.permutation(length) if self.shuffle else np.arange(length)
            idx[i * length : (i + 1) * length] = t
        return idx

    def __iter__(self):
        """Skip, because this is a private iterator called only from sampler."""
        return self

    def __next__(self) -> List[np.ndarray]:
        """Retrieve next elements."""
        if self.offset >= self.epochs * self.length:
            raise StopIteration

        end_idx = self.offset + self.batch_size
        if end_idx <= self.epochs * self.length:
            idx_slice = self.idx[self.offset : end_idx]
            res = [d[idx_slice] for d in self.data]
        else:
            if self.drop_last:
                raise StopIteration
            idx_slice = self.idx[self.offset :]
            res = [d[idx_slice] for d in self.data]
            last_batch = []
            for i in range(len(res)):
                shape = list(res[i].shape)
                shape[0] = self.batch_size
                v = np.full(shape, self.backfill[i], dtype=res[i].dtype)
                v[: len(res[i])] = res[i]
                last_batch.append(v)
            res = last_batch

        self.offset += self.data_parallel_num * self.batch_size

        if len(res) == 1:
            return res[0]
        return res


class FileEdgeSampler(BaseSampler):
    r"""
    Loads all files and generate edge samples.

    Edge Sample File format (default delimeter='\\t')
         - src_id\\tdst_id[\\tfeature_dim1\\tfeature_dim2...]
    """

    def __init__(
        self,
        edge_type: Union[int, List[int]],
        sample_files: str,
        batch_size: int,
        epochs: int = 1,
        shuffle: bool = False,
        delimeter: str = "\t",
        feature_dim: int = 0,
        feature_type: np.dtype = np.float32,
        drop_last: bool = False,
        backfill_id: int = INVALID_NODE_ID,
        worker_index: int = 0,
        num_workers: int = 1,
        data_parallel_num: int = 1,
        data_parallel_index: int = 0,
    ):
        r"""
        Initialize FileEdgeSampler.

        Args:
          edge_type(int): the edge type for selection.
          sample_files(str): filenames, support pathname pattern expansion `glob.glob(sample_files)`.
          batch_size(int): how many samples per batch to load.
          epochs(int, optional): how many epochs to repeat sample files. (default: 1).
          shuffle(int, optional): set to True to have the data reshuffled at every epoch (default: False).
          delimeter(str, optional): The string used to separate values. (default: '\\t').
          feature_dim(int, optional): the feature dimentions in sample files. (default: 0).
          feature_type(numpy.dtype, optional): the feature dimentions in sample files. (default: np.float32).
          drop_last(bool, optional): set to True to drop the last incomplete batch, if the dataset size is not divisible by the batch size. If False and the size of dataset is not divisible by the batch size, the last batch will be padded with backfill_id. (default: False)
          backfill_id(int, optional): backfill value for the last no-full batch. (default: INVALID_NODE_ID)
          worker_index(int, optional): worker index from distrbiuted training. for single worker job, please use default value. (default: 0)
          num_workers(int, optional): total number of workers for distrbiuted training. for single worker job, please use default value. (default: 1)
        """
        super(FileEdgeSampler, self).__init__(
            batch_size,
            epochs,
            shuffle=shuffle,
            sample_num=-1,
            data_parallel_index=data_parallel_index,
            data_parallel_num=data_parallel_num,
        )
        self.feature_dim = feature_dim
        self.feature_type = feature_type
        self.edge_type = edge_type
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.backfill_id = backfill_id
        filelist = get_files(sample_files, worker_index, num_workers)
        self._load_edge_files(filelist, delimeter)

    def _load_edge_files(self, filelist: List[str], delimeter: str):
        self.logger.info("Edge Sample files: {0}".format(", ".join(filelist)))
        edges = []
        features = []
        ftype = get_feature_type(self.feature_type)

        for f in filelist:
            for line in open(f):
                col = line.split(delimeter)
                edges.append([int(col[0]), int(col[1])])
                if self.feature_dim > 0:
                    assert self.feature_dim + 2 <= len(col)
                    features.append([ftype(i) for i in col[2:]])

        self.edges = np.array(edges, dtype=np.int64)
        self.edges = np.concatenate(
            [self.edges, np.full((len(edges), 1), self.edge_type)], axis=1
        )
        if len(features) > 0:
            self.features = np.array(features, dtype=self.feature_type)
        self.logger.info("total #edges: {0}".format(len(self.edges)))

    def __len__(self) -> int:
        """Total number of minibatches in this sampler."""
        if self.drop_last:
            return math.floor(
                len(self.edges)
                * self.epochs
                / (self.batch_size * self.data_parallel_num)
            )
        else:
            return math.ceil(
                len(self.edges)
                * self.epochs
                / (self.batch_size * self.data_parallel_num)
            )

    def __iter__(self) -> _NumpyIterator:
        """Create a numpy based iterator."""
        if self.feature_dim == 0:
            return _NumpyIterator(
                [self.edges],
                self.batch_size,
                self.epochs,
                shuffle=self.shuffle,
                backfill=[
                    np.array(
                        [self.backfill_id, self.backfill_id, self.edge_type],
                        dtype=np.int64,
                    )
                ],
                drop_last=self.drop_last,
                data_parallel_num=self.data_parallel_num,
                data_parallel_index=self.data_parallel_index,
            )

        return _NumpyIterator(
            [self.edges, self.features],
            self.batch_size,
            self.epochs,
            shuffle=self.shuffle,
            backfill=[
                np.array(
                    [self.backfill_id, self.backfill_id, self.edge_type], dtype=np.int64
                ),
                np.zeros(shape=(self.feature_dim), dtype=np.float32),
            ],
            data_parallel_num=self.data_parallel_num,
            data_parallel_index=self.data_parallel_index,
        )


class FileNodeSampler(BaseSampler):
    r"""
    Loading all files and generate node samples.

    Node Sample File format: (each line has one node id, linesep='\\n')
      - node_id\\n
    """

    def __init__(
        self,
        sample_files: str,
        batch_size: int,
        epochs: int = 1,
        shuffle: bool = False,
        drop_last: bool = False,
        backfill_id: int = INVALID_NODE_ID,
        worker_index: int = 0,
        num_workers: int = 1,
        data_parallel_num: int = 1,
        data_parallel_index: int = 0,
    ):
        """
        Initialize FileNodeSampler.

        Args:
          sample_files(str): filenames, support pathname pattern expansion `glob.glob(sample_files)`.
          batch_size(int): how many samples per batch to load.
          epochs(int): how many epochs to repeat sample files.
          shuffle(int, optional): set to True to have the data reshuffled at every epoch, shuffle is going to happen only inside one worker. (default: False).
          drop_last(bool, optional): set to True to drop the last incomplete batch, if the dataset size is not divisible by the batch size. If False and the size of dataset is not divisible by the batch size, the last batch will be padded with backfill_id. (default: False)
          backfill_id(int, optional): backfill value for the last no-full batch. (default: INVALID_NODE_ID)
          worker_index(int, optional): worker index from distrbiuted training. for single worker job, please use default value. (default: 0)
          num_workers(int, optional): total number of workers for distrbiuted training. for single worker job, please use default value. (default: 1)
        """
        super().__init__(
            batch_size,
            epochs=epochs,
            shuffle=shuffle,
            sample_num=-1,
            data_parallel_index=data_parallel_index,
            data_parallel_num=data_parallel_num,
        )
        self.drop_last = drop_last
        self.backfill_id = backfill_id
        filelist = get_files(sample_files, worker_index, num_workers)
        self.logger.info("Node Sample files: {0}".format(", ".join(filelist)))
        self.nodes: np.ndarray = np.concatenate(
            [np.fromfile(f, dtype=np.int64, sep="\n") for f in filelist]
        )

    def __iter__(self) -> _NumpyIterator:
        """Return numpy iterator over file data."""
        return _NumpyIterator(
            data=[self.nodes],
            batch_size=self.batch_size,
            epochs=self.epochs,
            shuffle=self.shuffle,
            backfill=[self.backfill_id],
            drop_last=self.drop_last,
            data_parallel_num=self.data_parallel_num,
            data_parallel_index=self.data_parallel_index,
        )

    def __len__(self) -> int:
        """Total number of minibatches in this sampler."""
        if self.drop_last:
            return math.floor(
                len(self.nodes)
                * self.epochs
                / (self.batch_size * self.data_parallel_num)
            )
        else:
            return math.ceil(
                len(self.nodes)
                * self.epochs
                / (self.batch_size * self.data_parallel_num)
            )


class _RangeNodeIterator(_GEIterator):
    """Private iterable component to iterate a range."""

    def __init__(
        self,
        first: int,
        last: int,
        batch_size: int,
        num_workers: int,
        backfill_id: int,
        data_parallel_num: int = 1,
        data_parallel_index: int = 0,
    ):
        """Initialize range node iterator."""
        self.first = first + data_parallel_index
        self.last = last
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.backfill_id = backfill_id
        self.data_parallel_index = data_parallel_index
        self.data_parallel_num = data_parallel_num

    def __next__(self) -> np.ndarray:
        """Create a new range and increment offset."""
        if self.first >= self.last:
            raise StopIteration

        last = self.first + self.batch_size * self.data_parallel_num
        res = np.arange(self.first, last, step=self.data_parallel_num)
        if last > self.last:
            res[(self.last - self.first) // self.data_parallel_num :] = self.backfill_id

        self.first += self.num_workers * self.batch_size * self.data_parallel_num
        return res


class RangeNodeSampler(BaseSampler):
    """Sampler to iterate a range as node id list."""

    def __init__(
        self,
        first: int,
        last: int,
        batch_size: int,
        worker_index: int,
        num_workers: int,
        backfill_id: int,
        data_parallel_num: int = 1,
        data_parallel_index: int = 0,
    ):
        """Initialize node sampler."""
        super().__init__(
            batch_size,
            epochs=1,
            shuffle=False,
            sample_num=last - first,
            data_parallel_index=data_parallel_index,
            data_parallel_num=data_parallel_num,
        )
        self.start_id = first + worker_index * batch_size
        self.last = last
        assert self.start_id < self.last
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.backfill_id = backfill_id

    def __iter__(self) -> _RangeNodeIterator:
        """Return a new range iterator."""
        return _RangeNodeIterator(
            self.start_id,
            self.last,
            self.batch_size,
            self.num_workers,
            self.backfill_id,
            data_parallel_num=self.data_parallel_num,
            data_parallel_index=self.data_parallel_index,
        )

    def __len__(self) -> int:
        """Total number of minibatches in this sampler."""
        res = math.ceil(
            (self.last - self.start_id)
            / (self.batch_size * self.num_workers * self.data_parallel_num)
        )
        return res


class _RangeEdgeIterator(_RangeNodeIterator):
    """Private component to iterate a range of edge ids."""

    def __init__(
        self,
        edge_type,
        first: int,
        last: int,
        batch_size: int,
        num_workers: int,
        backfill_id: int,
        data_parallel_num: int = 1,
        data_parallel_index: int = 0,
    ):
        super().__init__(
            first,
            last,
            batch_size,
            num_workers,
            backfill_id,
            data_parallel_index=data_parallel_index,
            data_parallel_num=data_parallel_num,
        )
        assert batch_size % 2 == 0
        self.edge_type_arr = np.full(
            (self.batch_size // 2, 1), edge_type, dtype=np.int64
        )

    def __next__(self) -> np.ndarray:
        # generate node list, shape (2 * batch_size,)
        nodes = super().__next__()
        nodes = nodes.reshape(-1, 2)
        edges = np.concatenate([nodes, self.edge_type_arr], axis=1)
        # edges.shape (batch_size, 3)
        return edges


class RangeEdgeSampler(RangeNodeSampler):
    """RangeEdgeSampler will generate edges from np.arange(start_id, end_id)."""

    def __init__(
        self,
        edge_type: Union[int, List[int]],
        first: int,
        last: int,
        batch_size: int,
        worker_index: int,
        num_workers: int,
        backfill_id: int,
        data_parallel_num: int = 1,
        data_parallel_index: int = 0,
    ):
        """Initialize edge sampler."""
        super().__init__(
            first,
            last,
            batch_size * 2,
            worker_index,
            num_workers,
            backfill_id,
            data_parallel_index=data_parallel_index,
            data_parallel_num=data_parallel_num,
        )
        self.edge_type = edge_type

    def __iter__(self) -> _RangeEdgeIterator:
        """Return a new edge iterator."""
        return _RangeEdgeIterator(
            self.edge_type,
            self.start_id,
            self.last,
            self.batch_size,
            self.num_workers,
            self.backfill_id,
            data_parallel_num=self.data_parallel_num,
            data_parallel_index=self.data_parallel_index,
        )


class CSVNodeSampler(BaseSampler):
    """Sampler to iterate node list from CSV file."""

    def __init__(
        self,
        batch_size: int,
        sample_file: str,
        epoch: int = 1,
        data_parallel_num: int = 1,
        data_parallel_index: int = 0,
    ):
        """Initialize node sampler."""
        super().__init__(
            batch_size,
            epoch,
            shuffle=False,
            data_parallel_index=data_parallel_index,
            data_parallel_num=data_parallel_num,
        )
        self.batch_size = batch_size
        node_list = []
        with open(sample_file, "r") as f:
            data_file = csv.reader(f)
            for _, d in enumerate(data_file):
                if len(d) != 0:
                    node_list.append([int(col) for col in d])

        self.node_list: np.ndarray = np.array(node_list)
        self.cur_batch = 0
        self.count = (
            (len(node_list) + self.batch_size - 1)
            // (self.batch_size * self.data_parallel_num)
            * epoch
        )

    def __iter__(self):
        """No-op."""
        return self

    def __next__(self) -> np.ndarray:
        """Implement iterator interface."""
        start_pos = (
            self.cur_batch * self.batch_size * self.data_parallel_num
            + self.data_parallel_index
        )
        if start_pos >= len(self.node_list):
            raise StopIteration
        self.cur_batch += 1
        end_pos = (
            self.cur_batch * self.batch_size * self.data_parallel_num
            + self.data_parallel_index
        )
        if end_pos >= len(self.node_list):
            end_pos = len(self.node_list)
        return np.array(self.node_list[start_pos : end_pos : self.data_parallel_num])

    def __len__(self) -> int:
        """Total number of minibatches in this sampler."""
        return self.count


class FileTupleSampler(BaseSampler):
    r"""
    Loading all files and generate tuple samples.

    File Tuple format (each line):
      - node_id\\tnode_type
    """

    def __init__(
        self,
        filename: str,
        batch_size: int,
        epochs: int = 1,
        shuffle: bool = False,
        drop_last: bool = False,
        backfill: int = INVALID_NODE_ID,
        worker_index: int = 0,
        num_workers: int = 1,
        data_parallel_num: int = 1,
        data_parallel_index: int = 0,
    ):
        """
        Initialize FileTupleSampler.

        Args:
          filename(str): filename.
          batch_size(int): how many samples per batch to load.
          epochs(int, optional): how many epochs to repeat sample files. (default: 1)
          shuffle(int, optional): set to True to have the data reshuffled at every epoch, shuffle is going to happen only inside one worker. (default: False).
          drop_last(bool, optional): set to True to drop the last incomplete batch, if the dataset size is not divisible by the batch size. If False and the size of dataset is not divisible by the batch size, the last batch will be padded with backfill_id. (default: False)
          backfill(int, optional): backfill value for the last no-full batch. (default: INVALID_NODE_ID)
          worker_index(int, optional): worker index from distrbiuted training. for single worker job, please use default value. (default: 0)
          num_workers(int, optional): total number of workers for distrbiuted training. for single worker job, please use default value. (default: 1)
        """
        super().__init__(
            batch_size,
            epochs=epochs,
            shuffle=shuffle,
            sample_num=-1,
            data_parallel_index=data_parallel_index,
            data_parallel_num=data_parallel_num,
        )
        self.logger.info("Sample file: {0}".format(filename))
        self.drop_last = drop_last
        self.backfill = backfill
        self.data = self._load_tuple_file(filename, worker_index, num_workers)

    @staticmethod
    def _tuple_parse_func(line: str) -> Tuple[int, int]:
        cols = line.split("\t")
        assert len(cols) == 2
        nid, ntype = int(cols[0]), int(cols[1])
        return nid, ntype

    def _load_tuple_file(self, filename, worker_index, num_workers):
        parse_func = FileTupleSampler._tuple_parse_func
        data = []
        self.logger.info("Load tuple file: {}".format(filename))
        for cnt, l in enumerate(open(filename)):
            if cnt % num_workers == worker_index:
                p = parse_func(l)
                data.append([p[0], p[1]])
        data = np.array(data, dtype=np.int64)
        self.logger.info("total #tuple: {}".format(len(data)))
        return data

    def __len__(self) -> int:
        """Total number of minibatches in this sampler."""
        if self.drop_last:
            return math.floor(
                len(self.data)
                * self.epochs
                / (self.batch_size * self.data_parallel_num)
            )
        else:
            return math.ceil(
                len(self.data)
                * self.epochs
                / (self.batch_size * self.data_parallel_num)
            )

    def __iter__(self) -> _NumpyIterator:
        """Create a new numpy based iterator from this sampler."""
        return _NumpyIterator(
            [self.data],
            self.batch_size,
            self.epochs,
            shuffle=self.shuffle,
            backfill=[np.array([self.backfill, self.backfill], dtype=np.int64)],
            drop_last=self.drop_last,
            data_parallel_num=self.data_parallel_num,
            data_parallel_index=self.data_parallel_index,
        )


class TextFileSampler(BaseSampler):
    """Sampler to iterate line-based text files which are in local filesystem or azure data lake gen1 (adl://)."""

    def __init__(
        self,
        store_name: str,
        filename: str,
        adl_config: str = None,
        batch_size: int = 512,
        buffer_size: int = 1024,
        epochs: int = 1,
        shuffle: bool = False,
        drop_last: bool = False,
        worker_index: int = 0,
        num_workers: int = 1,
        read_block_in_M: int = 50,
        buffer_queue_size: int = 3,
        data_parallel_num: int = 1,
        data_parallel_index: int = 0,
    ):
        """Initialize sampler."""
        super().__init__(
            batch_size,
            epochs=epochs,
            shuffle=shuffle,
            sample_num=-1,
            data_parallel_index=data_parallel_index,
            data_parallel_num=data_parallel_num,
        )
        self.file_iter = TextFileIterator(
            adl_config=adl_config,
            store_name=store_name,
            filename=filename,
            batch_size=batch_size,
            epochs=epochs,
            read_block_in_M=read_block_in_M,
            buffer_queue_size=buffer_queue_size,
            shuffle=shuffle,
            drop_last=drop_last,
            worker_index=(worker_index * self.data_parallel_num)
            + self.data_parallel_index,
            num_workers=num_workers * self.data_parallel_num,
        )

        # a buffer which use to random sampling
        self.buf = []  # type: ignore
        min_buf_size = batch_size * 3
        self.buffer_size = buffer_size if buffer_size >= min_buf_size else min_buf_size
        if buffer_size < min_buf_size:
            self.logger.warn(
                f"buffer size {buffer_size} specified is smaller than the minimum {min_buf_size}, using minimum buffer size {min_buf_size} instead."
            )
        self.end = False

    def __len__(self) -> int:
        """Raise error, because data is streamed."""
        raise NotImplementedError

    def __iter__(self):
        """Start iteration from beginning."""
        self.reset()
        return self

    def __del__(self):
        """Stop iteration."""
        self.file_iter.join()

    def __next__(self) -> List[str]:
        """Load next elements from file."""
        if self.end and len(self.buf) == 0:
            raise StopIteration

        new_added = False
        while not self.end and len(self.buf) < self.buffer_size:
            try:
                batch = next(self.file_iter)
                self.buf.extend(batch)
                new_added = True
            except StopIteration:
                self.end = True
                break

        if self.shuffle and new_added:
            random.shuffle(self.buf)

        count = self.batch_size if len(self.buf) >= self.batch_size else len(self.buf)
        batch = self.buf[0:count]
        self.buf = self.buf[count:]
        while len(batch) != self.batch_size:
            batch.append("")

        return batch

    def reset(self):
        """Reset iterator and clear buffers."""
        self.file_iter.reset()
        self.buf = []
        self.end = False


def _node_label_feature_parser_func(
    line: str,
) -> Tuple[np.int64, np.float32, np.ndarray]:
    r"""
    Tab separated lines.

    Format:
     src_id\tlabel\tfeatures
     features: feature_1 feature_2 ...
     example: 10123\t1.0\t0.5 0.2 0.34 0.3
     - src_id: 10123
     - label: 1.0
     - feature: 0.5 0.2 0.34 0.3
    """
    cols = line.split("\t")
    assert len(cols) == 3
    nid = np.int64(int(cols[0]))
    label = np.float32(float(cols[1]))
    feature = np.array([float(x) for x in cols[2].split(" ")], np.float32)
    return nid, label, feature


class FileTupleSamplerV2(BaseSampler):
    """FileTupleSamplerV2 loads all files and generate tuple samples."""

    def __init__(
        self,
        filename: str,
        batch_size: int,
        epochs: int = 1,
        shuffle: bool = False,
        drop_last: bool = False,
        backfill: int = INVALID_NODE_ID,
        worker_index: int = 0,
        num_workers: int = 1,
        line_parser_func: Callable[
            [str], Tuple[np.int64, np.int32, np.ndarray]
        ] = _node_label_feature_parser_func,
        data_parallel_num: int = 1,
        data_parallel_index: int = 0,
    ):
        """
        Initialize FileTupleSamplerV2.

        Args:
          filename(str): filename.
          batch_size(int): how many samples per batch to load.
          epochs(int, optional): how many epochs to repeat sample files. (default: 1)
          shuffle(int, optional): set to True to have the data reshuffled at every epoch, shuffle is going to happen only inside one worker. (default: False).
          drop_last(bool, optional): set to True to drop the last incomplete batch, if the dataset size is not divisible by the batch size. If False and the size of dataset is not divisible by the batch size, the last batch will be padded with backfill_id. (default: False)
          backfill(int, optional): backfill value for the last no-full batch. (default: INVALID_NODE_ID)
          worker_index(int, optional): worker index from distrbiuted training. for single worker job, please use default value. (default: 0)
          num_workers(int, optional): total number of workers for distrbiuted training. for single worker job, please use default value. (default: 1)
        """
        super().__init__(
            batch_size,
            epochs=epochs,
            shuffle=shuffle,
            sample_num=-1,
            data_parallel_index=data_parallel_index,
            data_parallel_num=data_parallel_num,
        )
        self.logger.info("Sample file: {0}".format(filename))
        self.drop_last = drop_last
        self.backfill = backfill
        self.data = self._load_tuple_file(
            filename, worker_index, num_workers, line_parser_func
        )

    def _load_tuple_file(
        self,
        filename: str,
        worker_index: int,
        num_workers: int,
        parse_func: Callable[[str], tuple],
    ) -> List[np.ndarray]:
        raw_data: List[List[tuple]] = []
        self.logger.info("Load tuple file: {}".format(filename))
        for cnt, l in enumerate(open(filename)):
            if cnt % num_workers == worker_index:
                parts = parse_func(l)
                if len(raw_data) == 0:
                    raw_data = [[] for _ in parts]
                if len(raw_data) != len(parts):
                    raise ValueError(
                        f"current line has {len(parts)} parts, which is different with others {len(raw_data)}"
                    )
                for i, p in enumerate(parts):
                    raw_data[i].append(p)

        data = [np.array(x) for x in raw_data]
        self.logger.info(
            f"tuple file has {len(data)} columns, and {len(data[0])} lines"
        )
        return data

    def __len__(self) -> int:
        """Return number of minibatches in this sampler."""
        if self.drop_last:
            return math.floor(
                len(self.data[0])
                * self.epochs
                / (self.batch_size * self.data_parallel_num)
            )
        else:
            return math.ceil(
                len(self.data[0])
                * self.epochs
                / (self.batch_size * self.data_parallel_num)
            )

    def __iter__(self) -> _NumpyIterator:
        """Create a new iterator."""
        return _NumpyIterator(
            self.data,
            self.batch_size,
            self.epochs,
            shuffle=self.shuffle,
            backfill=[self.backfill] * len(self.data),
            drop_last=self.drop_last,
            data_parallel_num=self.data_parallel_num,
            data_parallel_index=self.data_parallel_index,
        )
