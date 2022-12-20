# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import pytest
import re
import random
import numpy as np
import urllib.request
import zipfile
import tempfile
from itertools import islice

num_a = 28646
num_p = 21044
num_v = 18


@pytest.fixture(scope="module")
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


@pytest.fixture(scope="module")
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


@pytest.fixture(scope="module")
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

    a_a_list_train_f = open(str(temp_data_dirname.join("a_a_list_train.txt")), "w")
    a_a_list_test_f = open(str(temp_data_dirname.join("a_a_list_test.txt")), "w")
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
    embed_f = open(model_path + "node_embedding.txt", "r")
    for line in islice(embed_f, 0, None):
        line = line.strip()
        node_id = re.split(" ", line)[0]
        index = int(node_id) - 1000000
        embed = np.asarray(re.split(" ", line)[1:], dtype="float32")
        a_embed[index] = embed
    embed_f.close()

    train_num = 0
    a_a_list_train_f = open(str(temp_data_dirname.join("a_a_list_train.txt")), "r")
    a_a_list_train_feature_f = open(
        str(temp_data_dirname.join("train_feature.txt")), "w"
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
    a_a_list_test_f = open(str(temp_data_dirname.join("a_a_list_test.txt")), "r")
    a_a_list_test_feature_f = open(str(temp_data_dirname.join("test_feature.txt")), "w")
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
    embed_f = open(model_path + "node_embedding.txt", "r")
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

    cluster_f = open(str(tmpdir.join("cluster.txt")), "w")
    cluster_embed_f = open(str(tmpdir.join("cluster_embed.txt")), "w")
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

    a_class_train_f = open(str(tmpdir.join("a_class_train.txt")), "w")
    a_class_test_f = open(str(tmpdir.join("a_class_test.txt")), "w")
    train_class_feature_f = open(str(tmpdir.join("train_class_feature.txt")), "w")
    test_class_feature_f = open(str(tmpdir.join("test_class_feature.txt")), "w")

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
