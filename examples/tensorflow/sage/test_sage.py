# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import random
import json
import tempfile
import logging
import pytest
import os
import sys
import networkx as nx
import numpy as np
from deepgnn.tf.common.utils import run_commands, load_embeddings
from deepgnn.tf.common.test_helper import TestHelper
from deepgnn.graph_engine.data.ppi import PPI

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
                ],
                "1": [cluster_id],
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
            "neighbor": {
                "0": dict(
                    [
                        (str(neighbor_id), 1.0)
                        for neighbor_id in nx.neighbors(g, node_id)
                    ]
                )
            },
        }
        data += json.dumps(node) + "\n"
        nodes.append(node)

    meta = '{"node_type_num": 1, \
             "node_float_feature_num": 2, \
             "node_binary_feature_num": 0, \
             "node_uint64_feature_num": 0, \
             "edge_type_num": 1, \
             "edge_float_feature_num": 0, \
             "edge_binary_feature_num": 0, \
             "edge_uint64_feature_num": 0}'

    raw_file = working_dir.name + "/data.json"
    with open(raw_file, "w+") as f:
        f.write(data)
    meta_file = working_dir.name + "/meta.json"
    with open(meta_file, "w+") as f:
        f.write(meta)

    convert.MultiWorkersConverter(
        graph_path=raw_file,
        meta_path=meta_file,
        partition_count=1,
        output_dir=working_dir.name,
        decoder_type=decoders.DecoderType.JSON,
    ).convert()

    logger.info(working_dir.name)
    logger.info(raw_file)
    logger.info(meta_file)


def setup_test(main_file):
    tmp_dir = tempfile.TemporaryDirectory()
    current_dir = os.path.dirname(os.path.realpath(__file__))
    mainfile = os.path.join(current_dir, main_file)
    return tmp_dir, tmp_dir.name, tmp_dir.name, mainfile


def get_f1_micro(model_dir):
    f1_micro = TestHelper.get_tf2_summary_value(
        os.path.join(model_dir, "evaluate/worker_0"), "f1_micro"
    )
    return f1_micro[-1]


def run_graphsage_supervised(data_dir, agg_type, epochs):
    tmp_dir, model_dir, _, mainfile = setup_test("main.py")

    def run_training_job():
        train_nodes = os.path.join(data_dir, "train.nodes")
        training_cmd = (
            f"python {mainfile} --gpu --eager --seed 123"
            + f" --model_dir {model_dir} --data_dir {data_dir}"
            + f" --mode train --node_files {train_nodes}"
            + f" --epochs {epochs} --batch_size 512"
            + " --neighbor_edge_types 0 --num_samples 25,10"
            + " --feature_idx 1 --feature_dim 50 --label_idx 0 --label_dim 121"
            + " --layer_dims 128,128 --num_classes 121"
            + f" --loss_name sigmoid --agg_type {agg_type} --backend snark --converter local"
        )
        res = run_commands(training_cmd)
        assert res == 0

    def run_eval_job(filename):
        val_nodes = os.path.join(data_dir, filename)
        eval_cmd = (
            f"python {mainfile} --gpu --eager --seed 123"
            + f" --model_dir {model_dir} --data_dir {data_dir}"
            + f" --mode evaluate --node_files {val_nodes}"
            + " --neighbor_edge_types 0,1 --num_samples 25,10"
            + " --feature_idx 1 --feature_dim 50 --label_idx 0 --label_dim 121"
            + " --layer_dims 128,128 --num_classes 121"
            + f" --loss_name sigmoid --agg_type {agg_type}"
            + " --log_save_steps 1 --summary_save_steps 1 --backend snark --converter local"
        )
        res = run_commands(eval_cmd)
        assert res == 0
        f1_micro = get_f1_micro(model_dir)
        logger.info(f"F1Score ({val_nodes}): f1 micro {f1_micro} - {agg_type}")
        return f1_micro

    def run_inf_job():
        train_nodes = os.path.join(data_dir, "train.nodes")
        inf_cmd = (
            f"python {mainfile} --gpu --eager --seed 123"
            + f" --model_dir {model_dir} --data_dir {data_dir}"
            + f" --mode inference --node_files {train_nodes}"
            + " --neighbor_edge_types 0,1 --num_samples 25,10"
            + " --feature_idx 1 --feature_dim 50 --label_idx 0 --label_dim 121"
            + " --layer_dims 128,128 --num_classes 121"
            + f" --loss_name sigmoid --agg_type {agg_type}"
            + " --log_save_steps 1 --summary_save_steps 1 --backend snark --converter local"
        )
        res = run_commands(inf_cmd)
        assert res == 0

    run_training_job()

    eval_f1_micro = run_eval_job("val.nodes")
    test_f1_micro = run_eval_job("test.nodes")

    run_inf_job()
    emb = load_embeddings(model_dir, 44906, 121)
    assert np.sum(np.linalg.norm(emb, axis=1) > 0) == 44906

    return eval_f1_micro, test_f1_micro


@pytest.mark.flaky(reruns=5)
def test_sage_supervised():
    g = PPI()
    data_dir = g.data_dir()
    eval_f1, test_f1 = run_graphsage_supervised(data_dir, agg_type="mean", epochs=15)
    logger.info(f"graphsage ppi - mean eval_f1:{eval_f1}, test_f1:{test_f1}")
    assert eval_f1 >= 0.582909741 - 0.0051  # f1 > average_f1 - stddev
    assert test_f1 >= 0.592567404 - 0.0054  # f1 > average_f1 - stddev


def test_sage_supervised_other_agg():
    g = PPI()
    data_dir = g.data_dir()
    # MaxPool: 10 epoch results
    # eval_f1 >= 0.594302915 - 0.0098 # f1 > average_f1 - stddev
    # test_f1 >= 0.605199247 - 0.0096 # f1 > average_f1 - stddev
    run_graphsage_supervised(data_dir, agg_type="maxpool", epochs=1)

    # LSTM : 10 epoch results
    # eval_f1 >= 0.597000355 - 0.0065 # f1 > average_f1 - stddev
    # test_f1 >= 0.607637893 - 0.0064 # f1 > average_f1 - stddev
    run_graphsage_supervised(data_dir, agg_type="lstm", epochs=1)


def test_sage_supervised_identity_feature():
    tmp_dir, model_dir, data_dir, mainfile = setup_test("main.py")
    prepare_connected_caveman_graph(tmp_dir)

    def run_training_job():
        training_cmd = (
            f"python {mainfile} --gpu --eager --seed 123"
            + f" --model_dir {model_dir} --data_dir {data_dir}"
            + " --mode train --node_types 0 "
            + " --epochs 30 --batch_size 5 --learning_rate 0.01"
            + " --neighbor_edge_types 0 --num_samples 5,5"
            + " --identity_feature True --all_node_count 50 --identity_dim 8 --label_idx 1 --label_dim 1"
            + " --layer_dims 8,8 --num_classes 5"
            + " --loss_name softmax --agg_type mean --backend snark --converter skip"
        )
        res = run_commands(training_cmd)
        assert res == 0

    def run_inf_job():
        inf_file = os.path.join(tmp_dir.name, "inf_file.nodes")
        with open(inf_file, "w") as fout:
            for i in range(50):
                fout.write(f"{i}\n")

        inf_cmd = (
            f"python {mainfile} --gpu --eager --seed 123"
            + f" --model_dir {model_dir} --data_dir {data_dir}"
            + " --mode inference --node_files {inf_file}"
            + " --batch_size 4"
            + " --neighbor_edge_types 0 --num_samples 5,5"
            + " --identity_feature True --all_node_count 50 --identity_dim 8 --label_idx 1 --label_dim 1"
            + " --layer_dims 8,8 --num_classes 5"
            + " --loss_name softmax --agg_type mean --backend snark --converter skip"
        )
        res = run_commands(inf_cmd)
        assert res == 0

    run_training_job()
    run_inf_job()

    # evaluate embeddings.
    # * If node i, j are in the same cluster and node i, k are in different clusters,
    #   the node embedding distance(i,j) should be closer than distance(i,k).
    max_id = 49
    dim = 5
    emb = load_embeddings(model_dir, max_id + 1, dim)
    src = 5
    distance = [0] * (max_id + 1)
    for cid in range(5):
        for nid in range(10):
            nid = cid * 10 + nid
            distance[nid] = np.linalg.norm(emb[src] - emb[nid])

        t = ",".join(["{: .2f}".format(i) for i in distance[cid * 10 : cid * 10 + 10]])
        logger.info("distance ({}, cluster_{}) = [{}]".format(src, cid, t))

    avg_distance_in_cluster = sum(distance[0:10]) / 10
    avg_distance_acrosss_cluster = sum(distance[10:]) / 40

    assert avg_distance_in_cluster < avg_distance_acrosss_cluster


def test_sage_unsupervised():
    run_sage_unsupervised("--feature_idx 0 --feature_dim 2")
    run_sage_unsupervised(
        "--identity_feature True --identity_dim 4 --all_node_count 50"
    )


def run_sage_unsupervised(hyper_param):
    tmp_dir, model_dir, data_dir, mainfile = setup_test("main_unsup.py")
    prepare_connected_caveman_graph(tmp_dir)

    def run_training_job():
        training_cmd = (
            f"python {mainfile} --eager --mode train --model_dir {model_dir} --data_dir {data_dir}"
            + " --batch_size 5 --learning_rate 0.01 --epochs 30"
            + f" --node_types 0 {hyper_param}"
            + " --num_samples 5,5 --layer_dims 5,5"
            + " --neighbor_edge_types 0"
            + " --negative_node_types 0 --negative_num 5"
            + " --loss_name xent --agg_type mean --backend snark --converter skip"
        )
        res = run_commands(training_cmd)
        assert res == 0

    def run_inf_job():
        inf_file = os.path.join(tmp_dir.name, "inf_file.nodes")
        with open(inf_file, "w") as fout:
            for i in range(50):
                fout.write(f"{i}\n")

        inf_cmd = (
            f"python {mainfile} --eager --mode inference --model_dir {model_dir} --data_dir {data_dir}"
            + " --batch_size 4 --learning_rate 0.01 --epochs 30"
            + f" --node_files {inf_file} {hyper_param}"
            + " --num_samples 5,5 --layer_dims 5,5"
            + " --neighbor_edge_types 0"
            + " --negative_node_types 0 --negative_num 5"
            + " --loss_name xent --agg_type mean --backend snark --converter skip"
        )
        res = run_commands(inf_cmd)
        assert res == 0

    run_training_job()
    run_inf_job()

    # evaluate embeddings.
    # * If node i, j are in the same cluster and node i, k are in different clusters,
    #   the node embedding distance(i,j) should be closer than distance(i,k).
    num_nodes = 50
    dim = 5 * 2
    emb = load_embeddings(model_dir, num_nodes + 1, dim)
    src = 5
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
