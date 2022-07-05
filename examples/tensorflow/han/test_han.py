# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import random
import json
import tempfile
import logging
import glob
import sys
import pytest
import os
import subprocess
import time
import networkx as nx
import numpy as np
import deepgnn.graph_engine.snark.convert as convert
import deepgnn.graph_engine.snark.decoders as decoders

logging.basicConfig(
    level=logging.DEBUG,
    format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s",
)
logger = logging.getLogger()


def prepare_connected_caveman_graph(working_dir):
    random.seed(0)
    num_clusters = 5
    num_nodes_in_cluster = 10
    g = nx.connected_caveman_graph(num_clusters, num_nodes_in_cluster)
    nodes = []
    data = ""
    for node_id in g:
        cluster_id = float(node_id // num_nodes_in_cluster)
        normalized_cluster = cluster_id / num_clusters - 0.4
        node = {
            "node_weight": 1,
            "node_id": node_id,
            "node_type": 0,
            "uint64_feature": None,
            "float_feature": {
                "0": [
                    0.02 * random.uniform(0, 1) + 2.5 * normalized_cluster - 0.01,
                    random.uniform(0, 1),
                ]
            },
            "binary_feature": None,
            "edge": [
                {
                    "src_id": node_id,
                    "dst_id": neighbor_id,
                    "edge_type": 0,
                    "weight": 1.0,
                }
                for neighbor_id in nx.neighbors(g, node_id)
            ],
        }
        data += json.dumps(node) + "\n"
        nodes.append(node)

    meta = '{"node_type_num": 1, \
             "node_float_feature_num": 1, \
             "node_binary_feature_num": 0, \
             "node_uint64_feature_num": 0, \
             "edge_type_num": 1, \
             "edge_float_feature_num": 0, \
             "edge_binary_feature_num": 0, \
             "edge_uint64_feature_num": 0}'

    raw_file = working_dir + "/data.json"
    with open(raw_file, "w+") as f:
        f.write(data)
    meta_file = working_dir + "/meta.json"
    with open(meta_file, "w+") as f:
        f.write(meta)

    convert.MultiWorkersConverter(
        graph_path=raw_file,
        meta_path=meta_file,
        partition_count=1,
        output_dir=working_dir,
        decoder=decoders.JsonDecoder(),
    ).convert()

    logger.info(working_dir)
    logger.info(raw_file)
    logger.info(meta_file)


def run_commands(commands):
    logger.info(commands)
    cmd_args = commands.strip().split(" ")
    proc = subprocess.Popen(args=cmd_args)
    while proc.poll() is None:
        logger.info("wait...")
        time.sleep(3)
    return proc.poll()


def load_embeddings(model_dir, num_nodes, dim):
    res = np.empty((num_nodes, dim), dtype=np.float32)
    files = glob.glob(os.path.join(model_dir, "embedding_*.tsv"))
    for fname in files:
        logger.info("load: {}".format(fname))
        for line in open(fname):
            col = line.split("\t")
            nid = int(col[0])
            emb = [float(c) for c in col[1].split(" ")]
            assert len(emb) == dim
            res[nid] = emb
    return res


class TempDir:
    def __init__(self, name):
        self.name = name


@pytest.mark.xfail(sys.platform == "win32", reason="flaky in windows, mark it as xfail")
def test_han():
    tmp_dir = tempfile.TemporaryDirectory()
    prepare_connected_caveman_graph(tmp_dir.name)

    model_dir = tmp_dir.name
    data_dir = tmp_dir.name
    current_dir = os.path.dirname(os.path.realpath(__file__))
    mainfile = os.path.join(current_dir, "main.py")

    # start training
    training_cmd = (
        "python {} --mode train --graph_type local --model_dir {} --data_dir {} --seed 123"
        + " --training_node_types 0 --edge_types 0 --num_nodes 100 --feature_idx 0 --feature_dim 2 --partitions 0"
        + " --batch_size 8 --learning_rate 0.001 --epochs 50 --head_num 1 --hidden_dim 16 --fanouts 5 --backend snark --converter skip"
    ).format(mainfile, model_dir, data_dir)
    res = run_commands(training_cmd)
    assert res == 0

    # start inference
    inference_cmd = (
        "python {} --mode inference --graph_type local --model_dir {} --data_dir {} --seed 123"
        + " --edge_types 0 --num_nodes 50 --feature_idx 0 --feature_dim 2 --batch_size 8"
        + " --head_num 1 --hidden_dim 16 --fanouts 5 --backend snark --converter skip"
    ).format(mainfile, model_dir, data_dir)
    res = run_commands(inference_cmd)
    assert res == 0

    # evaluate
    num_nodes = 50
    dim = 16
    emb = load_embeddings(model_dir, num_nodes + 1, dim)
    src = 1
    distance = [0] * (num_nodes + 1)
    for cid in range(5):
        for nid in range(10):
            nid = cid * 10 + nid
            distance[nid] = np.linalg.norm(emb[src] - emb[nid])

        t = ",".join(["{: .2f}".format(i) for i in distance[cid * 10 : cid * 10 + 10]])
        logger.info("distance ({}, cluster_{}) = [{}]".format(src, cid, t))

    avg_distance_in_cluster = sum(distance[0:10]) / 10
    avg_distance_acrosss_cluster = sum(distance[10:]) / 40

    assert avg_distance_in_cluster < avg_distance_acrosss_cluster

    tmp_dir.cleanup()


if __name__ == "__main__":
    sys.exit(
        pytest.main(
            [__file__, "--junitxml", os.environ["XML_OUTPUT_FILE"], *sys.argv[1:]]
        )
    )
