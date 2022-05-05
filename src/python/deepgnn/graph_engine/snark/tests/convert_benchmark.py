# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import json
import os
from time import time
import struct
import sys
import tempfile
from pathlib import Path

import pytest
import numpy as np
import numpy.testing as npt

import deepgnn.graph_engine.snark.convert as convert
from deepgnn.graph_engine.snark.decoders import DecoderType, json_to_linear, dst_sort
from deepgnn.graph_engine.snark.dispatcher import QueueDispatcher


def generate_json(folder, N, fan_out):
    data = open(os.path.join(folder, "graph.json"), "w+")
    dst_ids = np.random.randint(0, N, N)
    for i in range(N):
        dst_id = int(dst_ids[i])

        edges = [
            {
                "src_id": i,
                "dst_id": (dst_id + ii) % N,
                "edge_type": 0,
                "weight": 0.5,
                "uint64_feature": {"0": [1, 2, 3, 4, 5]},
                "float_feature": {},
                "binary_feature": {},
            }
            for ii in range(fan_out)
        ]

        node = {
            "node_id": i,
            "node_type": int(i > (N // 2)),
            "node_weight": 1,
            "neighbor": {
                "0": {str((dst_id + ii) % N): 0.5 for ii in range(fan_out)},
                "1": {},
            },
            "uint64_feature": {},
            "float_feature": {
                "0": [0, 1, 2, 3, 4, 5],
                "1": [-0.01, -0.02, 0.3, 0.4, 0.5],
            },
            "binary_feature": {},
            "edge": edges,
        }
        json.dump(node, data)
        data.write("\n")
    data.flush()

    meta = open(os.path.join(folder, "meta.txt"), "w+")
    meta.write(
        '{"node_type_num": 3, "edge_type_num": 2, \
        "node_uint64_feature_num": 0, "node_float_feature_num": 2, \
        "node_binary_feature_num": 0, "edge_uint64_feature_num": 1, \
        "edge_float_feature_num": 1, "edge_binary_feature_num": 1}'
    )
    meta.flush()

    data.close()
    meta.close()

    return data.name, meta.name


def benchmark_json_to_linear(data_name, meta_name, linear_name):
    time_start = time()
    json_to_linear(data_name, linear_name)
    return time() - time_start


def benchmark_to_binary(data_name, meta_name, output_dir, decoder_type, **kw):
    from deepgnn.graph_engine.snark import dispatcher

    dispatcher.PROCESS_PRINT_INTERVAL = 10**10
    time_start = time()
    convert.MultiWorkersConverter(
        graph_path=data_name,
        meta_path=meta_name,
        partition_count=1,
        output_dir=output_dir,
        decoder_type=decoder_type,
        **kw,
    ).convert()
    return time() - time_start


def benchmark_json_to_binary(data_name, meta_name, output_dir):
    return benchmark_to_binary(data_name, meta_name, output_dir, DecoderType.JSON)


def benchmark_linear_to_binary(data_name, meta_name, output_dir):
    # buffer_size reduces number of TextFileIterator q.get calls == json, record_per_step reduces number TextFileIterator.__next__ == json
    return benchmark_to_binary(data_name, meta_name, output_dir, DecoderType.LINEAR, buffer_size=50 // 4, record_per_step=512 * 21)


if __name__ == "__main__":
    input_dir = "/tmp/convert_benchmark"
    json_binary_dir = "/tmp/json_binary_dir"
    linear_binary_dir = "/tmp/linear_binary_dir"

    linear_name = os.path.join(input_dir, "graph.linear")

    data_name, meta_name = generate_json(input_dir, 10**5, 20)#5 * 10**4, 20)
    print(
        f"JSON to Linear: {benchmark_json_to_linear(data_name, meta_name, linear_name)}"
    )

    import cProfile, pstats
    from pstats import SortKey

    profile = True
    if profile:
        pr = cProfile.Profile()
        pr.enable()

    print(f"JSON to Binary: {benchmark_json_to_binary(data_name, meta_name, json_binary_dir)}")

    if profile:
        pr.disable()
        ps = pstats.Stats(pr)
        ps.strip_dirs()
        ps.sort_stats(SortKey.CUMULATIVE) #.print_stats()
        ps.dump_stats("/home/user/DeepGNN/profile_json")

        pr = cProfile.Profile()
        pr.enable()

    print(f"Linear to Binary: {benchmark_linear_to_binary(linear_name, meta_name, linear_binary_dir)}")

    if profile:
        pr.disable()
        ps = pstats.Stats(pr)
        ps.strip_dirs()
        ps.sort_stats(SortKey.CUMULATIVE) #.print_stats()
        ps.dump_stats("/home/user/DeepGNN/profile")
