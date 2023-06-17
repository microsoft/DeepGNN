# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
import csv
import numpy as np
import os
import random
import tempfile
import time
import zipfile
import re
import random
import numpy as np
import urllib.request
import tempfile
from itertools import islice

from typing import Tuple

import torch
from torch.utils.data import IterableDataset

from deepgnn import get_logger
from contextlib import contextmanager
from deepgnn.graph_engine import (
    Graph,
    SamplingStrategy,
)
from model import HetGnnModel  # type: ignore
from sampler import HetGnnDataSampler  # type: ignore
import evaluation  # type: ignore

node_base_index = 1000000

num_a = 28646
num_p = 21044
num_v = 18

logger = get_logger()


@contextmanager
def prepare_local_test_files():
    name = "academic.zip"
    working_dir = tempfile.TemporaryDirectory()
    zip_file = os.path.join(working_dir.name, name)
    urllib.request.urlretrieve(
        f"https://deepgraphpub.blob.core.windows.net/public/testdata/{name}", zip_file
    )
    with zipfile.ZipFile(zip_file, "r") as zip_ref:
        zip_ref.extractall(working_dir.name)

    yield working_dir.name
    working_dir.cleanup()


def load_data(prepare_local_test_files):
    a_net_embed = {}
    p_net_embed = {}
    v_net_embed = {}
    net_e_f = open(
        os.path.join(prepare_local_test_files, "academic", "node_net_embedding.txt"),
        "r",
    )
    for line in islice(net_e_f, 1, None):
        line = line.strip()
        index = re.split(" ", line)[0]
        if len(index) and (index[0] == "a" or index[0] == "v" or index[0] == "p"):
            embeds = np.asarray(re.split(" ", line)[1:], dtype="float32")
            if index[0] == "a":
                a_net_embed[index[1:]] = embeds
            elif index[0] == "v":
                v_net_embed[index[1:]] = embeds
            else:
                p_net_embed[index[1:]] = embeds
    net_e_f.close()
    features = [a_net_embed, p_net_embed, v_net_embed]

    a_p_list_train = [[] for k in range(num_a)]
    p_a_list_train = [[] for k in range(num_p)]
    p_p_cite_list_train = [[] for k in range(num_a)]
    v_p_list_train = [[] for k in range(num_v)]

    relation_f = [
        "a_p_list_train.txt",
        "p_a_list_train.txt",
        "p_p_citation_list.txt",
        "v_p_list_train.txt",
    ]

    # store academic relational data
    for i in range(len(relation_f)):
        f_name = relation_f[i]
        neigh_f = open(os.path.join(prepare_local_test_files, "academic", f_name), "r")
        for line in neigh_f:
            line = line.strip()
            node_id = int(re.split(":", line)[0])
            neigh_list = re.split(":", line)[1]
            neigh_list_id = re.split(",", neigh_list)
            if f_name == "a_p_list_train.txt":
                for j in range(len(neigh_list_id)):
                    a_p_list_train[node_id].append("p" + str(neigh_list_id[j]))
            elif f_name == "p_a_list_train.txt":
                for j in range(len(neigh_list_id)):
                    p_a_list_train[node_id].append("a" + str(neigh_list_id[j]))
            elif f_name == "p_p_citation_list.txt":
                for j in range(len(neigh_list_id)):
                    p_p_cite_list_train[node_id].append("p" + str(neigh_list_id[j]))
            else:
                for j in range(len(neigh_list_id)):
                    v_p_list_train[node_id].append("p" + str(neigh_list_id[j]))
        neigh_f.close()

    # store paper venue
    p_v = [0] * num_p
    p_v_f = open(os.path.join(prepare_local_test_files, "academic", "p_v.txt"), "r")
    for line in p_v_f:
        line = line.strip()
        p_id = int(re.split(",", line)[0])
        v_id = int(re.split(",", line)[1])
        p_v[p_id] = v_id
    p_v_f.close()

    # paper neighbor: author + citation + venue
    p_neigh_list_train = [[] for k in range(num_p)]
    for i in range(num_p):
        p_neigh_list_train[i] += p_a_list_train[i]
        p_neigh_list_train[i] += p_p_cite_list_train[i]
        p_neigh_list_train[i].append("v" + str(p_v[i]))

    adj_list = [a_p_list_train, p_neigh_list_train, v_p_list_train]

    return features, adj_list, prepare_local_test_files


def init_het_input_data(prepare_local_test_files):
    a_p_list_train = [[] for k in range(num_a)]
    a_p_list_test = [[] for k in range(num_a)]
    p_a_list_train = [[] for k in range(num_p)]
    p_a_list_test = [[] for k in range(num_p)]
    p_p_cite_list_train = [[] for k in range(num_p)]
    p_p_cite_list_test = [[] for k in range(num_p)]
    v_p_list_train = [[] for k in range(num_v)]

    relation_f = [
        "a_p_list_train.txt",
        "a_p_list_test.txt",
        "p_a_list_train.txt",
        "p_a_list_test.txt",
        "p_p_cite_list_train.txt",
        "p_p_cite_list_test.txt",
        "v_p_list_train.txt",
    ]

    # store academic relational data
    for i in range(len(relation_f)):
        f_name = relation_f[i]
        neigh_f = open(os.path.join(prepare_local_test_files, "academic", f_name), "r")
        for line in neigh_f:
            line = line.strip()
            node_id = int(re.split(":", line)[0])
            neigh_list = re.split(":", line)[1]
            neigh_list_id = re.split(",", neigh_list)
            if f_name == "a_p_list_train.txt":
                for j in range(len(neigh_list_id)):
                    a_p_list_train[node_id].append("p" + str(neigh_list_id[j]))
            elif f_name == "a_p_list_test.txt":
                for j in range(len(neigh_list_id)):
                    a_p_list_test[node_id].append("p" + str(neigh_list_id[j]))
            elif f_name == "p_a_list_train.txt":
                for j in range(len(neigh_list_id)):
                    p_a_list_train[node_id].append("a" + str(neigh_list_id[j]))
            elif f_name == "p_a_list_test.txt":
                for j in range(len(neigh_list_id)):
                    p_a_list_test[node_id].append("a" + str(neigh_list_id[j]))
            elif f_name == "p_p_cite_list_train.txt":
                for j in range(len(neigh_list_id)):
                    p_p_cite_list_train[node_id].append("p" + str(neigh_list_id[j]))
            elif f_name == "p_p_cite_list_test.txt":
                for j in range(len(neigh_list_id)):
                    p_p_cite_list_test[node_id].append("p" + str(neigh_list_id[j]))
            else:
                for j in range(len(neigh_list_id)):
                    v_p_list_train[node_id].append("p" + str(neigh_list_id[j]))
        neigh_f.close()

    # store paper venue
    p_v = [0] * num_p
    p_v_f = open(os.path.join(prepare_local_test_files, "academic", "p_v.txt"), "r")
    for line in p_v_f:
        line = line.strip()
        p_id = int(re.split(",", line)[0])
        v_id = int(re.split(",", line)[1])
        p_v[p_id] = v_id
    p_v_f.close()

    # paper neighbor: author + citation + venue
    p_neigh_list_train = [[] for k in range(num_p)]
    for i in range(num_p):
        p_neigh_list_train[i] += p_a_list_train[i]
        p_neigh_list_train[i] += p_p_cite_list_train[i]
        p_neigh_list_train[i].append("v" + str(p_v[i]))

    return {
        "a_p_list_train": a_p_list_train,
        "a_p_list_test": a_p_list_test,
        "p_a_list_train": p_a_list_train,
        "p_a_list_test": p_a_list_test,
        "p_p_cite_list_train": p_p_cite_list_train,
        "p_p_cite_list_test": p_p_cite_list_test,
        "p_neigh_list_train": p_neigh_list_train,
        "v_p_list_train": v_p_list_train,
        "p_v": p_v,
    }


def a_a_collaborate_train_test(args, model_path, input_data_map, temp_data_dirname):
    a_a_list_train = [[] for k in range(args.A_n)]
    a_a_list_test = [[] for k in range(args.A_n)]
    p_a_list = [input_data_map["p_a_list_train"], input_data_map["p_a_list_test"]]

    for t in range(len(p_a_list)):
        for i in range(len(p_a_list[t])):
            for j in range(len(p_a_list[t][i])):
                for k in range(j + 1, len(p_a_list[t][i])):
                    if t == 0:
                        a_a_list_train[int(p_a_list[t][i][j][1:])].append(
                            int(p_a_list[t][i][k][1:])
                        )
                        a_a_list_train[int(p_a_list[t][i][k][1:])].append(
                            int(p_a_list[t][i][j][1:])
                        )
                    else:
                        if len(a_a_list_train[int(p_a_list[t][i][j][1:])]) and len(
                            a_a_list_train[int(p_a_list[t][i][k][1:])]
                        ):
                            if (
                                int(p_a_list[t][i][k][1:])
                                not in a_a_list_train[int(p_a_list[t][i][j][1:])]
                            ):
                                a_a_list_test[int(p_a_list[t][i][j][1:])].append(
                                    int(p_a_list[t][i][k][1:])
                                )
                            if (
                                int(p_a_list[t][i][j][1:])
                                not in a_a_list_train[int(p_a_list[t][i][k][1:])]
                            ):
                                a_a_list_test[int(p_a_list[t][i][k][1:])].append(
                                    int(p_a_list[t][i][j][1:])
                                )

    for i in range(args.A_n):
        a_a_list_train[i] = list(set(a_a_list_train[i]))
        a_a_list_test[i] = list(set(a_a_list_test[i]))

    a_a_list_train_f = open(str(temp_data_dirname + "/a_a_list_train.txt"), "w")
    a_a_list_test_f = open(str(temp_data_dirname + "/a_a_list_test.txt"), "w")
    a_a_list = [a_a_list_train, a_a_list_test]

    for t in range(len(a_a_list)):
        for i in range(len(a_a_list[t])):
            # print (i)
            if len(a_a_list[t][i]):
                if t == 0:
                    for j in range(len(a_a_list[t][i])):
                        a_a_list_train_f.write(
                            "%d, %d, %d\n" % (i, a_a_list[t][i][j], 1)
                        )
                        node_n = random.randint(0, args.A_n - 1)
                        while node_n in a_a_list[t][i]:
                            node_n = random.randint(0, args.A_n - 1)
                        a_a_list_train_f.write("%d, %d, %d\n" % (i, node_n, 0))
                else:
                    for j in range(len(a_a_list[t][i])):
                        a_a_list_test_f.write(
                            "%d, %d, %d\n" % (i, a_a_list[t][i][j], 1)
                        )
                        node_n = random.randint(0, args.A_n - 1)
                        while (
                            node_n in a_a_list[t][i]
                            or node_n in a_a_list_train[i]
                            or len(a_a_list_train[i]) == 0
                        ):
                            node_n = random.randint(0, args.A_n - 1)
                        a_a_list_test_f.write("%d, %d, %d\n" % (i, node_n, 0))

    a_a_list_train_f.close()
    a_a_list_test_f.close()

    return a_a_collab_feature_setting(args, model_path, temp_data_dirname)


def a_a_collab_feature_setting(args, model_path, temp_data_dirname):
    a_embed = np.around(np.random.normal(0, 0.01, [args.A_n, args.embed_d]), 4)
    embed_f = open(model_path + "/node_embedding.txt", "r")
    for line in islice(embed_f, 0, None):
        line = line.strip()
        node_id = re.split(" ", line)[0]
        index = int(node_id) - 1000000
        embed = np.asarray(re.split(" ", line)[1:], dtype="float32")
        a_embed[index] = embed
    embed_f.close()

    train_num = 0
    a_a_list_train_f = open(str(temp_data_dirname + "/a_a_list_train.txt"), "r")
    a_a_list_train_feature_f = open(
        str(temp_data_dirname + "/train_feature.txt"), "w"
    )
    for line in a_a_list_train_f:
        line = line.strip()
        a_1 = int(re.split(",", line)[0])
        a_2 = int(re.split(",", line)[1])

        label = int(re.split(",", line)[2])
        if random.random() < 0.2:  # training data ratio
            train_num += 1
            a_a_list_train_feature_f.write("%d, %d, %d," % (a_1, a_2, label))
            for d in range(args.embed_d - 1):
                a_a_list_train_feature_f.write(
                    "%f," % (a_embed[a_1][d] * a_embed[a_2][d])
                )
            a_a_list_train_feature_f.write(
                "%f" % (a_embed[a_1][args.embed_d - 1] * a_embed[a_2][args.embed_d - 1])
            )
            a_a_list_train_feature_f.write("\n")
    a_a_list_train_f.close()
    a_a_list_train_feature_f.close()

    test_num = 0
    a_a_list_test_f = open(str(temp_data_dirname + "/a_a_list_test.txt"), "r")
    a_a_list_test_feature_f = open(str(temp_data_dirname + "/test_feature.txt"), "w")
    for line in a_a_list_test_f:
        line = line.strip()
        a_1 = int(re.split(",", line)[0])
        a_2 = int(re.split(",", line)[1])

        label = int(re.split(",", line)[2])
        test_num += 1
        a_a_list_test_feature_f.write("%d, %d, %d," % (a_1, a_2, label))
        for d in range(args.embed_d - 1):
            a_a_list_test_feature_f.write("%f," % (a_embed[a_1][d] * a_embed[a_2][d]))
        a_a_list_test_feature_f.write(
            "%f" % (a_embed[a_1][args.embed_d - 1] * a_embed[a_2][args.embed_d - 1])
        )
        a_a_list_test_feature_f.write("\n")
    a_a_list_test_f.close()
    a_a_list_test_feature_f.close()

    return train_num, test_num


def a_class_cluster_feature_setting(args, model_path, tmpdir, test_rootdir):
    a_embed = np.around(np.random.normal(0, 0.01, [args.A_n, args.embed_d]), 4)
    embed_f = open(model_path + "/node_embedding.txt", "r")
    for line in islice(embed_f, 0, None):
        line = line.strip()
        node_id = re.split(" ", line)[0]
        index = int(node_id) - 1000000
        embed = np.asarray(re.split(" ", line)[1:], dtype="float32")
        a_embed[index] = embed
    embed_f.close()

    a_p_list_train = [[] for k in range(args.A_n)]
    a_p_list_train_f = open(
        os.path.join(test_rootdir, "academic", "a_p_list_train.txt"), "r"
    )
    for line in a_p_list_train_f:
        line = line.strip()
        node_id = int(re.split(":", line)[0])
        neigh_list = re.split(":", line)[1]
        neigh_list_id = re.split(",", neigh_list)
        for j in range(len(neigh_list_id)):
            a_p_list_train[node_id].append("p" + str(neigh_list_id[j]))
    a_p_list_train_f.close()

    p_v = [0] * args.P_n
    p_v_f = open(os.path.join(test_rootdir, "academic", "p_v.txt"), "r")
    for line in p_v_f:
        line = line.strip()
        p_id = int(re.split(",", line)[0])
        v_id = int(re.split(",", line)[1])
        p_v[p_id] = v_id
    p_v_f.close()

    a_v_list_train = [[] for k in range(args.A_n)]
    for i in range(len(a_p_list_train)):  # tranductive node classification
        for j in range(len(a_p_list_train[i])):
            p_id = int(a_p_list_train[i][j][1:])
            a_v_list_train[i].append(p_v[p_id])

    a_v_num = [[0 for k in range(args.V_n)] for k in range(args.A_n)]
    for i in range(args.A_n):
        for j in range(len(a_v_list_train[i])):
            v_index = int(a_v_list_train[i][j])
            a_v_num[i][v_index] += 1

    a_max_v = [0] * args.A_n
    for i in range(args.A_n):
        a_max_v[i] = a_v_num[i].index(max(a_v_num[i]))

    cluster_f = open(str(tmpdir + "/cluster.txt"), "w")
    cluster_embed_f = open(str(tmpdir + "/cluster_embed.txt"), "w")
    a_class_list = [[] for k in range(args.C_n)]
    cluster_id = 0
    num_hidden = args.embed_d
    for i in range(args.A_n):
        if len(a_p_list_train[i]):
            if (
                a_max_v[i] == 17 or a_max_v[i] == 4 or a_max_v[i] == 1
            ):  # cv: cvpr, iccv, eccv
                a_class_list[0].append(i)
                cluster_f.write("%d,%d\n" % (cluster_id, 3))
                cluster_embed_f.write("%d " % (cluster_id))
                for k in range(num_hidden):
                    cluster_embed_f.write("%lf " % (a_embed[i][k]))
                cluster_embed_f.write("\n")
                cluster_id += 1
            elif (
                a_max_v[i] == 16 or a_max_v[i] == 2 or a_max_v[i] == 3
            ):  # nlp: acl, emnlp, naacl
                a_class_list[1].append(i)
                cluster_f.write("%d,%d\n" % (cluster_id, 0))
                cluster_embed_f.write("%d " % (cluster_id))
                for k in range(num_hidden):
                    cluster_embed_f.write("%lf " % (a_embed[i][k]))
                cluster_embed_f.write("\n")
                cluster_id += 1
            elif (
                a_max_v[i] == 9 or a_max_v[i] == 13 or a_max_v[i] == 6
            ):  # dm: kdd, wsdm, icdm
                a_class_list[2].append(i)
                cluster_f.write("%d,%d\n" % (cluster_id, 1))
                cluster_embed_f.write("%d " % (cluster_id))
                for k in range(num_hidden):
                    cluster_embed_f.write("%lf " % (a_embed[i][k]))
                cluster_embed_f.write("\n")
                cluster_id += 1
            elif (
                a_max_v[i] == 12 or a_max_v[i] == 10 or a_max_v[i] == 5
            ):  # db: sigmod, vldb, icde
                a_class_list[3].append(i)
                cluster_f.write("%d,%d\n" % (cluster_id, 2))
                cluster_embed_f.write("%d " % (cluster_id))
                for k in range(num_hidden):
                    cluster_embed_f.write("%lf " % (a_embed[i][k]))
                cluster_embed_f.write("\n")
                cluster_id += 1
    cluster_f.close()
    cluster_embed_f.close()

    a_class_train_f = open(str(tmpdir + ("/a_class_train.txt")), "w")
    a_class_test_f = open(str(tmpdir + ("/a_class_test.txt")), "w")
    train_class_feature_f = open(str(tmpdir + ("/train_class_feature.txt")), "w")
    test_class_feature_f = open(str(tmpdir + ("/test_class_feature.txt")), "w")

    train_num = 0
    test_num = 0
    for i in range(args.C_n):
        for j in range(len(a_class_list[i])):
            randvalue = random.random()
            if randvalue < 0.1:
                a_class_train_f.write("%d,%d\n" % (a_class_list[i][j], i))
                train_class_feature_f.write("%d,%d," % (a_class_list[i][j], i))
                for d in range(num_hidden - 1):
                    train_class_feature_f.write("%lf," % a_embed[a_class_list[i][j]][d])
                train_class_feature_f.write(
                    "%lf" % a_embed[a_class_list[i][j]][num_hidden - 1]
                )
                train_class_feature_f.write("\n")
                train_num += 1
            else:
                a_class_test_f.write("%d,%d\n" % (a_class_list[i][j], i))
                test_class_feature_f.write("%d,%d," % (a_class_list[i][j], i))
                for d in range(num_hidden - 1):
                    test_class_feature_f.write("%lf," % a_embed[a_class_list[i][j]][d])
                test_class_feature_f.write(
                    "%lf" % a_embed[a_class_list[i][j]][num_hidden - 1]
                )
                test_class_feature_f.write("\n")
                test_num += 1
    a_class_train_f.close()
    a_class_test_f.close()

    return train_num, test_num, cluster_id


class MockHetGnnFileNodeLoader(IterableDataset):
    def __init__(
        self, graph: Graph, batch_size: int = 200, sample_file: str = "", model=None
    ):
        self.graph = graph
        self.batch_size = batch_size
        node_list = []
        with open(sample_file, "r") as f:
            data_file = csv.reader(f)
            for i, d in enumerate(data_file):
                node_list.append([int(d[0]) + node_base_index, int(d[1])])
        self.node_list = np.array(node_list)
        self.cur_batch = 0
        self.model = model

    def __iter__(self):
        """Implement IterableDataset method to provide data iterator."""
        return self

    def __next__(self):
        """Implement iterator interface."""
        if self.cur_batch * self.batch_size >= len(self.node_list):
            raise StopIteration
        start_pos = self.cur_batch * self.batch_size
        self.cur_batch += 1
        end_pos = self.cur_batch * self.batch_size
        if end_pos >= len(self.node_list):
            end_pos = -1
        context = {}
        context["inputs"] = np.array(self.node_list[start_pos:end_pos][:, 0])
        context["node_type"] = self.node_list[start_pos:end_pos][0][1]
        context["encoder"] = self.model.build_node_context(
            context["inputs"], self.graph
        )
        return context


class MockGraph:
    def __init__(self, feat_data, adj_lists):
        self.feat_data = feat_data
        self.adj_lists = adj_lists
        self.type_ranges = [
            (node_base_index, node_base_index + 2000),
            (node_base_index * 2, node_base_index * 2 + 2000),
            (node_base_index * 3, node_base_index * 3 + 10),
        ]

    def sample_nodes(
        self,
        size: int,
        node_type: int,
        strategy: SamplingStrategy = SamplingStrategy.Random,
    ) -> np.ndarray:
        return np.random.randint(
            self.type_ranges[node_type][0], self.type_ranges[node_type][1], size
        )

    def map_node_id(self, node_id, node_type):
        if node_type == "a" or node_type == "0":
            return int(node_id) + node_base_index
        if node_type == "p" or node_type == "1":
            return int(node_id) + (node_base_index * 2)
        if node_type == "v" or node_type == "2":
            return int(node_id) + (node_base_index * 3)

    def sample_neighbors(
        self,
        nodes: np.ndarray,
        edge_types: np.ndarray,
        count: int = 10,
        strategy: str = "byweight",
        default_node: int = -1,
        default_weight: float = 0.0,
        default_node_type: int = -1,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        res = np.empty((len(nodes), count), dtype=np.int64)
        res_types = np.full((len(nodes), count), -1, dtype=np.int32)
        res_count = np.empty((len(nodes)), dtype=np.int64)
        for i in range(len(nodes)):
            universe = []
            if nodes[i] != -1:
                node_type = (nodes[i] // node_base_index) - 1
                universe = [
                    self.map_node_id(x[1:], x[0])
                    for x in self.adj_lists[node_type][nodes[i] % node_base_index]
                ]

            # If there are no neighbors, fill results with a dummy value.
            if len(universe) == 0:
                res[i] = np.full(count, -1, dtype=np.int64)
                res_count[i] = 0
            else:
                res[i] = np.random.choice(universe, count, replace=True)
                res_count[i] = count

            for nt in range(len(res[i])):
                if res[i][nt] != -1:
                    neightype = (res[i][nt] // node_base_index) - 1
                    res_types[i][nt] = neightype

        return (
            res,
            np.full((len(nodes), count), 0.0, dtype=np.float32),
            res_types,
            res_count,
        )

    def node_features(
        self, nodes: np.ndarray, features: np.ndarray, feature_type: np.dtype
    ) -> np.ndarray:
        node_features = np.zeros((len(nodes), features[0][1]), dtype=np.float32)
        for i in range(len(nodes)):
            node_id = nodes[i]
            node_type = (node_id // node_base_index) - 1
            key = str(node_id % node_base_index)
            if node_id == -1 or key not in self.feat_data[node_type]:
                continue

            node_features[i] = self.feat_data[node_type][key][0 : features[0][1]]
        return node_features


def parse_testing_args(arg_str):
    parser = argparse.ArgumentParser(description="application data process")
    parser.add_argument("--A_n", type=int, default=28646, help="number of author node")
    parser.add_argument("--P_n", type=int, default=21044, help="number of paper node")
    parser.add_argument("--V_n", type=int, default=18, help="number of venue node")
    parser.add_argument("--C_n", type=int, default=4, help="number of node class label")
    parser.add_argument("--embed_d", type=int, default=128, help="embedding dimension")

    args = parser.parse_args(arg_str)
    return args


def parse_training_args(arg_str):
    parser = argparse.ArgumentParser(description="application data process")
    parser.add_argument(
        "--node_type", default=0, type=int, help="Node type to train/evaluate model."
    )
    parser.add_argument("--batch_size", default=512, type=int, help="Mini-batch size.")
    parser.add_argument(
        "--num_epochs", default=10, type=int, help="Number of epochs for training."
    )
    parser.add_argument(
        "--neighbor_count",
        type=int,
        default=10,
        help="number of neighbors to sample of each node",
    )
    parser.add_argument("--walk_length", default=5, type=int)
    parser.add_argument(
        "--node_type_count",
        type=int,
        default=2,
        help="number of node type in the graph",
    )
    parser.add_argument(
        "--model_dir", type=str, default="./hettrain", help="path to save model"
    )
    parser.add_argument(
        "--save_model_freq",
        type=float,
        default=2,
        help="number of iterations to save model",
    )
    parser.add_argument("--cuda", default=0, type=int)
    parser.add_argument("--checkpoint", default="", type=str)
    parser.add_argument(
        "--learning_rate", default=0.01, type=float, help="Learning rate."
    )
    parser.add_argument("--dim", default=256, type=int, help="Dimension of embedding.")
    parser.add_argument("--max_id", type=int, help="Max node id.")
    parser.add_argument("--feature_idx", default=-1, type=int, help="Feature index.")
    parser.add_argument("--feature_dim", default=0, type=int, help="Feature dimension.")
    parser.add_argument(
        "--data_dir", type=str, default="", help="Local graph data dir."
    )
    parser.add_argument(
        "--sample_file",
        type=str,
        default="",
        help="file contains node id and type to calculate embeddings.",
    )

    args = parser.parse_args(arg_str)
    return args


def get_train_args(data_dir, model_dir, test_rootdir):
    args = parse_training_args(
        [
            "--data_dir=" + data_dir,
            "--neighbor_count=10",
            "--model_dir=" + model_dir,
            "--num_epochs=2",
            "--batch_size=128",
            "--walk_length=5",
            "--dim=128",
            "--max_id=1024",
            "--node_type_count=3",
            "--neighbor_count=10",
            "--feature_dim=128",
            "--sample_file="
            + os.path.join(test_rootdir, "academic", "a_node_list.txt"),
            "--feature_idx=0",
        ]
    )
    return args


class MockIterableDataset(torch.utils.data.IterableDataset):
    def __init__(self, batch_size, graph, model, sampler):
        self.graph = graph
        self.batch_size = batch_size
        self.model = model
        self.sampler = sampler

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        assert worker_info is None
        for inputs in self.sampler:
            result = self.model.query(self.graph, inputs)
            yield result


def train_academic_data(g, test_rootdir):
    torch.manual_seed(0)
    np.random.seed(0)

    model_path = tempfile.TemporaryDirectory()
    model_path_name = model_path.name + "/"

    args = get_train_args("", model_path_name, test_rootdir)

    # train model
    model = HetGnnModel(
        node_type_count=args.node_type_count,
        neighbor_count=args.neighbor_count,
        embed_d=args.dim,
        feature_type=np.float32,
        feature_idx=args.feature_idx,
        feature_dim=args.feature_dim,
    )

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.learning_rate,
        weight_decay=0,
    )

    for epoch in range(args.num_epochs):
        # reset dataset means we can iterate the dataset in next epoch
        ds = MockIterableDataset(
            sampler=HetGnnDataSampler(
                graph=g,
                num_nodes=args.batch_size,
                batch_size=args.batch_size,
                node_type_count=args.node_type_count,
            ),
            graph=g,
            model=model,
            batch_size=args.batch_size,
        )
        data_loader = torch.utils.data.DataLoader(ds, batch_size=None)

        logger.info("Epoch {}".format(epoch))
        times = []
        start_time = time.time()
        scores = []
        labels = []

        for i, context in enumerate(data_loader):
            optimizer.zero_grad()
            loss, score, label = model(context)
            scores.append(score)
            labels.append(label)
            loss.backward()
            optimizer.step()

            if i % 20 == 0:
                end_time = time.time()
                logger.info(
                    "step: {:04d}; loss: {:.4f}; time: {:.4f}s".format(
                        i, loss.data.item(), (end_time - start_time)
                    )
                )

                times.append(end_time - start_time)
                start_time = time.time()

        if epoch % args.save_model_freq == 0 or epoch == args.num_epochs - 1:
            logger.info("model saved in epoch: {:04d}".format(epoch))
            torch.save(model.state_dict(), model_path_name + "gnnmodel.pt")

        metric = model.compute_metric(scores, labels)
        logger.info("Mean epoch {}: {}".format(model.metric_name(), metric))

    # trained node embedding path
    return model_path, model


def save_embedding(model_path, graph, test_rootdir):
    args = get_train_args("", model_path, test_rootdir)
    model = HetGnnModel(
        node_type_count=args.node_type_count,
        neighbor_count=args.neighbor_count,
        embed_d=args.dim,
        feature_type=np.float32,
        feature_idx=args.feature_idx,
        feature_dim=args.feature_dim,
    )

    model.load_state_dict(torch.load(model_path + "/gnnmodel.pt"))
    model.train()

    embed_file = open(model_path+ "/node_embedding.txt", "w")

    batch_size = 200
    saving_dataset = MockHetGnnFileNodeLoader(
        graph=graph, batch_size=batch_size, model=model, sample_file=args.sample_file
    )

    data_loader = torch.utils.data.DataLoader(saving_dataset)
    for _, context in enumerate(data_loader):
        out_temp = model.get_embedding(context)
        out_temp = out_temp.data.cpu().numpy()
        inputs = context["inputs"].squeeze(0)
        for k in range(len(out_temp)):
            embed_file.write(
                str(inputs[k].numpy())
                + " "
                + " ".join([str(out_temp[k][x]) for x in range(len(out_temp[k]))])
                + "\n"
            )

    embed_file.close()


def test_link_prediction_on_het_gnn(model_path, input_data_map, tmpdir):  # noqa: F811
    random.seed(0)
    # input_data_map = init_het_input_data

    # do evaluation
    args = parse_testing_args([])
    train_num, test_num = a_a_collaborate_train_test(
        args, model_path, input_data_map, tmpdir
    )
    auc, f1 = evaluation.evaluate_link_prediction(args, train_num, test_num, tmpdir)

    assert auc > 0.6 and auc < 0.9
    assert f1 > 0.6 and f1 < 0.9


def test_classification_on_het_gnn(
    prepare_local_test_files, save_embedding, tmpdir  # noqa: F811
):
    random.seed(0)

    model_path = save_embedding

    # do evaluation
    args = parse_testing_args([])
    train_num, test_num, _ = a_class_cluster_feature_setting(
        args, model_path, tmpdir, prepare_local_test_files
    )
    macroF1, microF1 = evaluation.evaluate_node_classification(
        args, train_num, test_num, tmpdir
    )
    assert macroF1 > 0.9
    assert microF1 > 0.9


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate HetGNN model.")
    parser.add_argument(
        "model",
        choices=["link_prediction", "classification"],
        default="link_prediction",
        const="link_prediction",
        nargs="?",
        help="Task to perform model training on: classification or link prediction.",
    )
    args = parser.parse_args()
    with prepare_local_test_files() as local_test_files:
        feat_data, adj_lists, test_rootdir = load_data(local_test_files)
        print(f"Loaded data from {test_rootdir}")
        het_input_data = init_het_input_data(local_test_files)
        graph = MockGraph(feat_data, adj_lists)
        print("Graph loaded!")
        model_path, model = train_academic_data(graph, test_rootdir)
        print("model trained")
        save_embedding(model_path.name, graph, test_rootdir)
        print("embedings saved")
        if args.model == "link_prediction":
            print(f"Passing {test_rootdir}")
            test_link_prediction_on_het_gnn(
                model_path.name, het_input_data, test_rootdir
            )
            print("Tested link prediction")
        else:
            test_classification_on_het_gnn(
                model_path.name, het_input_data, test_rootdir
            )
