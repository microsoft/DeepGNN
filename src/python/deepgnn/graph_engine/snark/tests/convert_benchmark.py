# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import json
import os
from time import time
import numpy as np
import deepgnn.graph_engine.snark.convert as convert
from deepgnn.graph_engine.snark.decoders import JsonDecoder, LinearDecoder, json_to_linear


def generate_json(folder, N, fan_out):
    data = open(os.path.join(folder, "graph.json"), "w+")
    dst_ids = np.random.randint(0, N, N)
    features = list(range(FEATURE_LEN))
    for i in range(N):
        dst_id = int(dst_ids[i])

        edges = [
            {
                "src_id": i,
                "dst_id": (dst_id + ii) % N,
                "edge_type": 0,
                "weight": 0.5,
                "uint64_feature": {"0": features} if FEATURE_LEN else {},
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
            "float_feature": {"0": features, "1": features} if FEATURE_LEN else {},
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


def benchmark_to_binary(data_name, meta_name, output_dir, decoder_class, **kw):
    from deepgnn.graph_engine.snark import dispatcher

    dispatcher.PROCESS_PRINT_INTERVAL = 10**10
    time_start = time()
    convert.MultiWorkersConverter(
        graph_path=data_name,
        meta_path=meta_name,
        partition_count=1,
        output_dir=output_dir,
        decoder_class=decoder_class,
        **kw,
    ).convert()
    return time() - time_start


def benchmark_json_to_binary(data_name, meta_name, output_dir):
    return benchmark_to_binary(data_name, meta_name, output_dir, JsonDecoder)


def benchmark_linear_to_binary(data_name, meta_name, output_dir):
    # buffer_size reduces number of TextFileIterator q.get calls == json, record_per_step reduces number TextFileIterator.__next__ == json
    return benchmark_to_binary(data_name, meta_name, output_dir, LinearDecoder, buffer_size=50 // 4)#, record_per_step=512 * (EDGES + 1))


NODES = 1 * 10**5  # 5
EDGES = 100
FEATURE_LEN = 0

if __name__ == "__main__":
    input_dir = "/tmp/convert_benchmark2"
    json_binary_dir = "/tmp/json_binary_dir2"
    linear_binary_dir = "/tmp/linear_binary_dir2"

    linear_name = os.path.join(input_dir, "graph.linear")

    data_name = os.path.join(input_dir, "graph.json")
    meta_name = os.path.join(input_dir, "meta.txt")
    if False:
        data_name, meta_name = generate_json(input_dir, NODES, EDGES)

        print(
            f"JSON to Linear: {benchmark_json_to_linear(data_name, meta_name, linear_name)}"
        )

        #exit()
    import cProfile, pstats
    from pstats import SortKey

    profile = False

    if False:
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

    if profile:
        pr = cProfile.Profile()
        pr.enable()

    print(f"Linear to Binary: {benchmark_linear_to_binary(linear_name, meta_name, linear_binary_dir)}")

    if profile:
        pr.disable()
        ps = pstats.Stats(pr)
        ps.strip_dirs()
        ps.sort_stats(SortKey.CUMULATIVE) #.print_stats()
        ps.dump_stats("/home/user/DeepGNN/profile")
