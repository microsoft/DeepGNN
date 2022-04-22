# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import sys
import traceback
import numpy as np
import threading
import atexit
import concurrent.futures.thread

from typing import Callable, Tuple
from queue import Queue
from concurrent.futures import ThreadPoolExecutor
from deepgnn.graph_engine.samplers import BaseSampler
from deepgnn.graph_engine._base import Graph
from deepgnn.graph_engine._adl_reader import FetchDone
from deepgnn import get_logger
from threading import BoundedSemaphore, Event

DEFAULT_THREAD_POOL_QUEUE_SIZE = 20


class BoundedExecutor:
    def __init__(self, max_workers=None, bound=DEFAULT_THREAD_POOL_QUEUE_SIZE):
        if bound <= 0:
            bound = DEFAULT_THREAD_POOL_QUEUE_SIZE
        if max_workers is not None and max_workers > 0 and max_workers > bound:
            bound = max_workers

        self.semaphore = BoundedSemaphore(bound)
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.shutdown(wait=True)
        return False

    def submit(self, fn, *args, **kwargs):
        self.semaphore.acquire()
        try:
            future = self.executor.submit(fn, *args, **kwargs)
        except:
            self.semaphore.release()
            raise
        else:
            future.add_done_callback(lambda x: self.semaphore.release())
            return future

    def shutdown(self, wait=True):
        self.executor.shutdown(wait)


def remove_threadpool_exit_handler():
    # Issue: if main thread throw exception, trainer will never exits becuase prefetch threads are keep running and waiting.
    # ThreadPoolExecutor added an exit handler, which will wait until all threads finish.
    # for more details, see here: https://github.com/python/cpython/blob/7df32f844efed33ca781a016017eab7050263b90/Lib/concurrent/futures/thread.py#L26

    # Workaround: remove threadpool exit handler.

    atexit.unregister(concurrent.futures.thread._python_exit)


remove_threadpool_exit_handler()


def _produce_func(
    graph: Graph,
    sampler: BaseSampler,
    model_query_fn: Callable,
    outputs: Queue,
    max_workers: int,
    stop_event: Event = None,
    collate_fn: Callable = None,
):
    """
    Produce examples for prefetch.
    """

    with BoundedExecutor(max_workers=max_workers) as pool:

        def process(inputs):
            try:
                graph_tensor = model_query_fn(graph, inputs)
                outputs.put(graph_tensor)
            except:
                traceback.print_exc(file=sys.stdout)
                outputs.put(FetchDone(exception=traceback.format_exc()))
                pool.shutdown(wait=False)
                raise

        # Sometimes computing the length of the sampler is expensive such as get the line count of a huge file
        # in azure data lake. So here we make the generator iterable and stop iteration once it read all
        # the data in outputs queue. We do that by adding an extra queue item which is the count of the sampler
        # when iterating the generator, we got this count and use it to decide if terminate or not.
        count = 0
        try:
            for sample in sampler:
                if stop_event is not None and stop_event.is_set():
                    return
                count += 1
                pool.submit(process, sample)
            outputs.put(FetchDone(count))
        except:
            traceback.print_exc(file=sys.stdout)
            outputs.put(FetchDone(count, traceback.format_exc()))
            raise


def _produce(
    graph: Graph,
    sampler: BaseSampler,
    model_query_fn: Callable,
    outputs: Queue,
    max_workers: int,
    stop_event: Event = None,
    collate_fn: Callable = None,
):
    _produce_func(
        graph=graph,
        sampler=sampler,
        model_query_fn=model_query_fn,
        outputs=outputs,
        max_workers=max_workers,
        stop_event=stop_event,
        collate_fn=collate_fn,
    )


# TODO(alsamylk): rebalance load between workers with a global step count
# TODO(alsamylk): rebalance load inside a worker: optimize queues and queries.
class Generator:
    """
    Generates query results from the context by organizing a publisher/subscriber queue.
    """

    def __init__(
        self,
        graph: Graph,
        sampler: BaseSampler,
        model_query_fn: Callable,
        prefetch_size: int = 30,
        max_parallel: int = 10,
        max_workers: int = 1,
        collate_fn: Callable = None,
    ):
        self.count = 0
        self.index = 0
        self.outputs: Queue[Tuple[np.array]] = Queue(prefetch_size)
        self.max_workers = max_workers
        self.pool = ThreadPoolExecutor(max_workers=max_workers)
        self.workers = []  # type: ignore
        self.stop_evt = Event()
        self.produce = _produce
        self.graph = graph
        self.sampler = sampler
        self.model_query_fn = model_query_fn
        self.max_parallel = max_parallel
        self.collate_fn = collate_fn

        # Start pub/sub queues
        for i in range(self.max_workers):
            f = self.pool.submit(
                self.produce,
                self.graph,
                self.sampler,
                self.model_query_fn,
                self.outputs,
                self.max_parallel,
                self.stop_evt,
                self.collate_fn,
            )
            self.workers.append(f)

    def __iter__(self):
        return self

    def __del__(self):
        self.join()

    # parse the queue and get the "count" queue item and use it to decide terminate the iteration or not.
    def __next__(self):
        if self.count > 0 and self.index == self.count + 1:
            raise StopIteration

        item = self.outputs.get()
        self.outputs.task_done()
        self.index += 1
        if type(item) is FetchDone:
            if item.exception_info is not None:
                get_logger().error(item.exception_info)
                raise Exception(item.exception_info)
            self.count = item.count
            if self.index == self.count + 1:
                raise StopIteration

            item = self.outputs.get()
            self.outputs.task_done()
            self.index += 1

        return item

    def _drain_outputs(self):
        while self.outputs.qsize() > 0:
            self.outputs.get()
            self.outputs.task_done()

    def join(self):
        if not threading.main_thread().is_alive():
            self.pool.shutdown(wait=False)
        else:
            self.stop_evt.set()
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
