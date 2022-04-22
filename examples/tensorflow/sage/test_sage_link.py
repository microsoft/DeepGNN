# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import random, json, tempfile, logging
import os, sys, pytest
import networkx as nx
import numpy as np
from copy import deepcopy
from deepgnn.tf.common.utils import run_commands, load_embeddings
import deepgnn.graph_engine.snark.convert as convert
import deepgnn.graph_engine.snark.decoders as decoders


logging.basicConfig(
    level=logging.DEBUG,
    format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s",
)
logger = logging.getLogger()


def prepare_user_teams_graph(working_dir):
    random.seed(246)
    np.random.seed(4812)

    num_clusters = 2
    num_nodes_in_cluster = 10
    g = nx.connected_caveman_graph(num_clusters, num_nodes_in_cluster)

    teams = list(np.arange(20, 30))
    user_team_positives_dict = {}
    for i in range(0, 5):
        user_team_positives_dict[i] = random.sample(teams[:3], 3)

    for i in range(5, 10):
        user_team_positives_dict[i] = random.sample(teams[2:5], 2)
        # user_team_positives_dict[i]=teams[:5]

    # user_team_positives_dict = {}
    for i in range(10, 15):
        user_team_positives_dict[i] = random.sample(teams[5:8], 3)

    for i in range(15, 20):
        user_team_positives_dict[i] = random.sample(teams[7:10], 2)
        # user_team_positives_dict[i]=teams[5:10]

    team_user_positives = {}
    for team in teams:
        team_user_positives[team] = []

    for user, team_set in user_team_positives_dict.items():
        for team in team_set:
            team_user_positives[team].append(user)

    all_user_set = set(np.arange(0, 20))
    team_user_negatives = {}
    for team, positives in team_user_positives.items():
        team_user_negatives[team] = all_user_set.difference(set(positives))

    all_teams_set = set(teams)
    user_team_negatives_dict = {}
    for node, pos_teams in user_team_positives_dict.items():
        user_team_negatives_dict[node] = list(all_teams_set.difference(set(pos_teams)))

    all_team_dict = {}
    for team in teams:
        all_team_dict[str(team)] = 1.0

    all_uids = list(np.arange(0, 20))

    # creating the json file for the graph

    val = np.zeros([30])

    total_edges = 0
    nodes = []
    data = ""
    for node_id in g:
        feats = deepcopy(val)
        feats[node_id] = 1.0

        edge_list = []
        for neigh_id in nx.neighbors(g, node_id):
            temp_dict = {}
            temp_dict["src_id"] = node_id
            temp_dict["dst_id"] = neigh_id
            temp_dict["edge_type"] = int(0)
            temp_dict["weight"] = 0.6
            temp_dict["uint64_feature"] = None
            temp_dict["float_feature"] = {"0": [0]}
            temp_dict["binary_feature"] = None
            edge_list.append(temp_dict)

        for neigh_id in user_team_positives_dict[node_id]:
            temp_dict = {}
            temp_dict["src_id"] = int(node_id)
            temp_dict["dst_id"] = int(neigh_id)
            temp_dict["edge_type"] = int(1)
            temp_dict["weight"] = 1.0
            temp_dict["uint64_feature"] = None
            temp_dict["float_feature"] = {"0": [1]}
            temp_dict["binary_feature"] = None
            edge_list.append(temp_dict)
            total_edges += 1

        for neigh_id in user_team_negatives_dict[node_id]:
            temp_dict = {}
            temp_dict["src_id"] = int(node_id)
            temp_dict["dst_id"] = int(neigh_id)
            temp_dict["edge_type"] = int(1)
            temp_dict["weight"] = 1.0
            temp_dict["uint64_feature"] = None
            temp_dict["float_feature"] = {"0": [0]}
            temp_dict["binary_feature"] = None
            edge_list.append(temp_dict)
            total_edges += 1

        node = {
            "node_weight": 1,
            "node_id": node_id,
            "node_type": 0,
            "uint64_feature": None,
            "float_feature": None,
            "binary_feature": None,
            "edge": edge_list,
            "neighbor": {
                "0": dict(
                    [(str(neigh_id), 1.0) for neigh_id in nx.neighbors(g, node_id)]
                ),
                "1": all_team_dict,
            },
        }

        data += json.dumps(node) + "\n"
        nodes.append(node)

    # adding the reverse team edges
    for team in teams:

        feats = deepcopy(val)
        feats[team] = 1.0

        edge_list = []
        for node_id in team_user_positives[team]:
            temp_dict = {}
            temp_dict["src_id"] = int(team)
            temp_dict["dst_id"] = int(node_id)
            temp_dict["edge_type"] = int(2)
            temp_dict["weight"] = 1.0
            temp_dict["uint64_feature"] = None
            temp_dict["float_feature"] = {"0": [1]}
            temp_dict["binary_feature"] = None
            edge_list.append(temp_dict)

        for node_id in team_user_negatives[team]:
            temp_dict = {}
            temp_dict["src_id"] = int(team)
            temp_dict["dst_id"] = int(node_id)
            temp_dict["edge_type"] = int(2)
            temp_dict["weight"] = 1.0
            temp_dict["uint64_feature"] = None
            temp_dict["float_feature"] = {"0": [0]}
            temp_dict["binary_feature"] = None
            edge_list.append(temp_dict)

        node = {
            "node_weight": 1.0,
            "node_id": int(team),
            "node_type": int(1),
            "uint64_feature": None,
            "float_feature": None,
            "binary_feature": None,
            "edge": edge_list,
            "neighbor": {"0": dict([(str(u_id), 1.0) for u_id in all_uids])},
        }

        data += json.dumps(node) + "\n"
        nodes.append(node)
        # {"0":dict([(str(u_id),1.0) for u_id in all_uids])}

    meta = '{"node_float_feature_num": 0, \
             "edge_binary_feature_num": 0, \
             "edge_type_num": 3, \
             "edge_float_feature_num": 1, \
             "node_type_num": 2, \
             "node_uint64_feature_num": 0, \
             "node_binary_feature_num": 0, \
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
        decoder_type=decoders.DecoderType.JSON,
    ).convert()


def setup_test(main_file):
    tmp_dir = tempfile.TemporaryDirectory()
    current_dir = os.path.dirname(os.path.realpath(__file__))
    mainfile = os.path.join(current_dir, main_file)
    return tmp_dir, tmp_dir.name, tmp_dir.name, mainfile


def test_sage_linkprediction():
    tmp_dir, model_dir, data_dir, mainfile = setup_test("main_linkprediction.py")
    prepare_user_teams_graph(data_dir)

    def run_training_job():
        training_cmd = (
            f"python {mainfile} --gpu --eager --mode train --model_dir {model_dir} --data_dir {data_dir} --seed 123"
            + " --batch_size 10 --learning_rate 0.01 --epochs 10 --edge_types 1"
            + " --agg_type mean --layer_dims 16,16 --num_classes 2 --dropout 0 --identity_dim 8 --all_node_count 30"
            + " --neighbor_edge_types 0 --num_samples 5,5 --label_idx 0 --label_dim 1"
            + " --log_save_steps 50 --summary_save_steps 50 --backend snark --converter skip"
        )
        res = run_commands(training_cmd)
        assert res == 0

    def run_eval_job():
        eval_cmd = (
            f"python {mainfile} --gpu --eager --mode evaluate --model_dir {model_dir} --data_dir {data_dir} --seed 123"
            + " --batch_size 10 --edge_types 1"
            + " --agg_type mean --layer_dims 16,16 --num_classes 2 --dropout 0 --identity_dim 8 --all_node_count 30"
            + " --neighbor_edge_types 0 --num_samples 5,5 --label_idx 0 --label_dim 1"
            + " --log_save_steps 1 --summary_save_steps 1 --backend snark --converter skip"
        )
        res = run_commands(eval_cmd)
        assert res == 0

    def run_inf_job():
        dst_node_file = os.path.join(data_dir, "dst.nodes")
        with open(dst_node_file, "w") as f:
            for i in range(20, 30):
                f.write(f"{i}\n")
        inf_cmd = (
            f"python {mainfile} --gpu --eager --mode inference --model_dir {model_dir} --data_dir {data_dir} --seed 123"
            + f" --batch_size 10 --inference_node dst --inference_files {dst_node_file}"
            + " --agg_type mean --layer_dims 16,16 --num_classes 2 --dropout 0 --identity_dim 8 --all_node_count 30"
            + " --neighbor_edge_types 0 --num_samples 5,5 --label_idx 0 --label_dim 1"
            + " --log_save_steps 1 --summary_save_steps 1 --backend snark --converter skip"
        )
        res = run_commands(inf_cmd)
        assert res == 0

    run_training_job()
    run_eval_job()
    run_inf_job()

    dim = 16
    emb = load_embeddings(model_dir, 30, dim=dim, fileprefix="embedding_*.tsv")
    assert emb.shape == (30, dim)
    assert all(emb[0:20, 0] == 0)
    assert all(emb[20:30, 0] != 0)


if __name__ == "__main__":
    sys.exit(
        pytest.main(
            [__file__, "--junitxml", os.environ["XML_OUTPUT_FILE"], *sys.argv[1:]]
        )
    )
