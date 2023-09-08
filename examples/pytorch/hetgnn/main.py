# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import numpy as np
import os
import random
import tempfile
import time
import zipfile
import re
import random
import urllib.request
import tempfile
from itertools import islice

import torch

from deepgnn import get_logger
from contextlib import contextmanager
from model import HetGnnModel  # type: ignore
from sampler import HetGnnDataSampler  # type: ignore
import evaluation  # type: ignore
from graph import MockGraph, MockHetGnnFileNodeLoader  # type: ignore

logger = get_logger()


def mrr(
    scores: torch.Tensor, labels: torch.Tensor, rank_in_ascending_order: bool = False
) -> torch.Tensor:
    """Compute metric based on logit scores."""
    assert len(scores.shape) > 1
    assert scores.size() == labels.size()

    size = scores.shape[-1]
    if rank_in_ascending_order:
        scores = -1 * scores
    _, indices_of_ranks = torch.topk(scores, k=size)
    _, ranks = torch.topk(-indices_of_ranks, k=size)
    return torch.mean(
        torch.reciprocal(
            torch.matmul(ranks.float(), torch.transpose(labels, -2, -1).float()) + 1
        )
    )


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


def load_data(config):
    a_net_embed = {}
    p_net_embed = {}
    v_net_embed = {}
    with open(
        os.path.join(config["data_dir"], "academic", "node_net_embedding.txt"),
        "r",
    ) as net_e_f:
        for line in islice(net_e_f, 1, None):
            line = line.strip()
            index = re.split(" ", line)[0]
            if len(index) and (index[0] == "a" or index[0] == "v" or index[0] == "p"):
                embeds = np.asarray(re.split(" ", line)[1:], dtype=np.float32)
                if index[0] == "a":
                    a_net_embed[index[1:]] = embeds
                elif index[0] == "v":
                    v_net_embed[index[1:]] = embeds
                else:
                    p_net_embed[index[1:]] = embeds

    features = [a_net_embed, p_net_embed, v_net_embed]

    a_p_list_train = [[] for _ in range(config["A_n"])]
    p_a_list_train = [[] for _ in range(config["P_n"])]
    p_p_cite_list_train = [[] for _ in range(config["A_n"])]
    v_p_list_train = [[] for _ in range(config["V_n"])]

    relation_f = [
        "a_p_list_train.txt",
        "p_a_list_train.txt",
        "p_p_citation_list.txt",
        "v_p_list_train.txt",
    ]

    # store academic relational data
    for i in range(len(relation_f)):
        f_name = relation_f[i]
        with open(os.path.join(config["data_dir"], "academic", f_name), "r") as neigh_f:
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

    # store paper venue
    p_v = [0] * config["P_n"]
    with open(os.path.join(config["data_dir"], "academic", "p_v.txt"), "r") as p_v_f:
        for line in p_v_f:
            line = line.strip()
            p_id = int(re.split(",", line)[0])
            v_id = int(re.split(",", line)[1])
            p_v[p_id] = v_id

    # paper neighbor: author + citation + venue
    p_neigh_list_train = [[] for k in range(config["P_n"])]
    for i in range(config["P_n"]):
        p_neigh_list_train[i] += p_a_list_train[i]
        p_neigh_list_train[i] += p_p_cite_list_train[i]
        p_neigh_list_train[i].append("v" + str(p_v[i]))

    adj_list = [a_p_list_train, p_neigh_list_train, v_p_list_train]

    return features, adj_list


def init_het_input_data(config):
    a_p_list_train = [[] for _ in range(config["A_n"])]
    a_p_list_test = [[] for _ in range(config["A_n"])]
    p_a_list_train = [[] for _ in range(config["P_n"])]
    p_a_list_test = [[] for _ in range(config["P_n"])]
    p_p_cite_list_train = [[] for _ in range(config["P_n"])]
    p_p_cite_list_test = [[] for _ in range(config["P_n"])]
    v_p_list_train = [[] for _ in range(config["V_n"])]

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
        with open(os.path.join(config["data_dir"], "academic", f_name), "r") as neigh_f:
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

    # store paper venue
    p_v = [0] * config["P_n"]
    with open(os.path.join(config["data_dir"], "academic", "p_v.txt"), "r") as p_v_f:
        for line in p_v_f:
            line = line.strip()
            p_id = int(re.split(",", line)[0])
            v_id = int(re.split(",", line)[1])
            p_v[p_id] = v_id

    # paper neighbor: author + citation + venue
    p_neigh_list_train = [[] for k in range(config["P_n"])]
    for i in range(config["P_n"]):
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


def a_a_collaborate_train_test(config, input_data_map):
    a_a_list_train = [[] for _ in range(config["A_n"])]
    a_a_list_test = [[] for _ in range(config["A_n"])]
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

    for i in range(config["A_n"]):
        a_a_list_train[i] = list(set(a_a_list_train[i]))
        a_a_list_test[i] = list(set(a_a_list_test[i]))

    with open(
        osp.join(config["data_dir"], "a_a_list_train.txt"), "w"
    ) as a_a_list_train_f:
        with open(
            osp.join(config["data_dir"], "a_a_list_test.txt"), "w"
        ) as a_a_list_test_f:
            a_a_list = [a_a_list_train, a_a_list_test]
            for t in range(len(a_a_list)):
                for i in range(len(a_a_list[t])):
                    if len(a_a_list[t][i]):
                        if t == 0:
                            for j in range(len(a_a_list[t][i])):
                                a_a_list_train_f.write(
                                    "%d, %d, %d\n" % (i, a_a_list[t][i][j], 1)
                                )
                                node_n = random.randint(0, config["A_n"] - 1)
                                while node_n in a_a_list[t][i]:
                                    node_n = random.randint(0, config["A_n"] - 1)
                                a_a_list_train_f.write("%d, %d, %d\n" % (i, node_n, 0))
                        else:
                            for j in range(len(a_a_list[t][i])):
                                a_a_list_test_f.write(
                                    "%d, %d, %d\n" % (i, a_a_list[t][i][j], 1)
                                )
                                node_n = random.randint(0, config["A_n"] - 1)
                                while (
                                    node_n in a_a_list[t][i]
                                    or node_n in a_a_list_train[i]
                                    or len(a_a_list_train[i]) == 0
                                ):
                                    node_n = random.randint(0, config["A_n"] - 1)
                                a_a_list_test_f.write("%d, %d, %d\n" % (i, node_n, 0))

    return a_a_collab_feature_setting(config)


def a_a_collab_feature_setting(config):
    a_embed = np.around(np.random.normal(0, 0.01, [config["A_n"], config["dim"]]), 4)
    with open(osp.join(config["data_dir"], "node_embedding.txt"), "r") as embed_f:
        for line in islice(embed_f, 0, None):
            line = line.strip()
            node_id = re.split(" ", line)[0]
            index = int(node_id) - 1000000
            embed = np.asarray(re.split(" ", line)[1:], dtype=np.float32)
            a_embed[index] = embed

    train_num = 0
    with open(
        osp.join(config["data_dir"], "a_a_list_train.txt"), "r"
    ) as a_a_list_train_f:
        with open(
            osp.join(config["data_dir"], "train_feature.txt"), "w"
        ) as a_a_list_train_feature_f:
            for line in a_a_list_train_f:
                line = line.strip()
                a_1 = int(re.split(",", line)[0])
                a_2 = int(re.split(",", line)[1])

                label = int(re.split(",", line)[2])
                if random.random() < 0.2:  # training data ratio
                    train_num += 1
                    a_a_list_train_feature_f.write("%d, %d, %d," % (a_1, a_2, label))
                    for d in range(config["dim"] - 1):
                        a_a_list_train_feature_f.write(
                            "%f," % (a_embed[a_1][d] * a_embed[a_2][d])
                        )
                    a_a_list_train_feature_f.write(
                        "%f"
                        % (
                            a_embed[a_1][config["dim"] - 1]
                            * a_embed[a_2][config["dim"] - 1]
                        )
                    )
                    a_a_list_train_feature_f.write("\n")

    test_num = 0
    with open(
        osp.join(config["data_dir"], "a_a_list_test.txt"), "r"
    ) as a_a_list_test_f:
        with open(
            osp.join(config["data_dir"], "test_feature.txt"), "w"
        ) as a_a_list_test_feature_f:
            for line in a_a_list_test_f:
                line = line.strip()
                a_1 = int(re.split(",", line)[0])
                a_2 = int(re.split(",", line)[1])

                label = int(re.split(",", line)[2])
                test_num += 1
                a_a_list_test_feature_f.write("%d, %d, %d," % (a_1, a_2, label))
                for d in range(config["dim"] - 1):
                    a_a_list_test_feature_f.write(
                        "%f," % (a_embed[a_1][d] * a_embed[a_2][d])
                    )
                a_a_list_test_feature_f.write(
                    "%f"
                    % (
                        a_embed[a_1][config["dim"] - 1]
                        * a_embed[a_2][config["dim"] - 1]
                    )
                )
                a_a_list_test_feature_f.write("\n")

    return train_num, test_num


def a_class_cluster_feature_setting(config):
    a_embed = np.around(np.random.normal(0, 0.01, [config["A_n"], config["dim"]]), 4)
    with open(osp.join(config["data_dir"] + "node_embedding.txt"), "r") as embed_f:
        for line in islice(embed_f, 0, None):
            line = line.strip()
            node_id = re.split(" ", line)[0]
            index = int(node_id) - 1000000
            embed = np.asarray(re.split(" ", line)[1:], dtype="float32")
            a_embed[index] = embed

    a_p_list_train = [[] for k in range(config["A_n"])]
    with open(
        os.path.join(config["data_dir"], "academic", "a_p_list_train.txt"), "r"
    ) as a_p_list_train_f:
        for line in a_p_list_train_f:
            line = line.strip()
            node_id = int(re.split(":", line)[0])
            neigh_list = re.split(":", line)[1]
            neigh_list_id = re.split(",", neigh_list)
            for j in range(len(neigh_list_id)):
                a_p_list_train[node_id].append("p" + str(neigh_list_id[j]))

    p_v = [0] * config["P_n"]
    with open(os.path.join(config["data_dir"], "academic", "p_v.txt"), "r") as p_v_f:
        for line in p_v_f:
            line = line.strip()
            p_id = int(re.split(",", line)[0])
            v_id = int(re.split(",", line)[1])
            p_v[p_id] = v_id

    a_v_list_train = [[] for k in range(config["A_n"])]
    for i in range(len(a_p_list_train)):  # tranductive node classification
        for j in range(len(a_p_list_train[i])):
            p_id = int(a_p_list_train[i][j][1:])
            a_v_list_train[i].append(p_v[p_id])

    a_v_num = [[0 for k in range(config["V_n"])] for k in range(config["A_n"])]
    for i in range(config["A_n"]):
        for j in range(len(a_v_list_train[i])):
            v_index = int(a_v_list_train[i][j])
            a_v_num[i][v_index] += 1

    a_max_v = [0] * config["A_n"]
    for i in range(config["A_n"]):
        a_max_v[i] = a_v_num[i].index(max(a_v_num[i]))

    with open(osp.join(config["data_dir"], "cluster.txt"), "w") as cluster_f:
        with open(
            osp.join(config["data_dir"], "cluster_embed.txt"), "w"
        ) as cluster_embed_f:
            a_class_list = [[] for k in range(config["C_n"])]
            cluster_id = 0
            num_hidden = config["dim"]
            for i in range(config["A_n"]):
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

    with open(
        osp.join(config["data_dir"], "a_class_train.txt"), "w"
    ) as a_class_train_f:
        with open(
            osp.join(config["data_dir"], "a_class_test.txt"), "w"
        ) as a_class_test_f:
            with open(
                osp.join(config["data_dir"], "train_class_feature.txt"), "w"
            ) as train_class_feature_f:
                with open(
                    osp.join(config["data_dir"], "test_class_feature.txt"), "w"
                ) as test_class_feature_f:
                    train_num = 0
                    test_num = 0
                    for i in range(config["C_n"]):
                        for j in range(len(a_class_list[i])):
                            randvalue = random.random()
                            if randvalue < 0.1:
                                a_class_train_f.write(
                                    "%d,%d\n" % (a_class_list[i][j], i)
                                )
                                train_class_feature_f.write(
                                    "%d,%d," % (a_class_list[i][j], i)
                                )
                                for d in range(num_hidden - 1):
                                    train_class_feature_f.write(
                                        "%lf," % a_embed[a_class_list[i][j]][d]
                                    )
                                train_class_feature_f.write(
                                    "%lf" % a_embed[a_class_list[i][j]][num_hidden - 1]
                                )
                                train_class_feature_f.write("\n")
                                train_num += 1
                            else:
                                a_class_test_f.write(
                                    "%d,%d\n" % (a_class_list[i][j], i)
                                )
                                test_class_feature_f.write(
                                    "%d,%d," % (a_class_list[i][j], i)
                                )
                                for d in range(num_hidden - 1):
                                    test_class_feature_f.write(
                                        "%lf," % a_embed[a_class_list[i][j]][d]
                                    )
                                test_class_feature_f.write(
                                    "%lf" % a_embed[a_class_list[i][j]][num_hidden - 1]
                                )
                                test_class_feature_f.write("\n")
                                test_num += 1

    return train_num, test_num


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


def train_academic_data(g, config):
    torch.manual_seed(0)
    np.random.seed(0)

    # train model
    model = HetGnnModel(
        node_type_count=config["node_type_count"],
        neighbor_count=config["neighbor_count"],
        embed_d=config["dim"],
        feature_type=np.float32,
        feature_idx=config["feature_idx"],
        feature_dim=config["feature_dim"],
    )

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config["learning_rate"],
        weight_decay=0,
    )

    for epoch in range(config["num_epochs"]):
        # reset dataset, it means we can iterate the dataset in next epoch
        ds = MockIterableDataset(
            sampler=HetGnnDataSampler(
                graph=g,
                num_nodes=config["batch_size"],
                batch_size=config["batch_size"],
                node_type_count=config["node_type_count"],
            ),
            graph=g,
            model=model,
            batch_size=config["batch_size"],
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
        scores = torch.unsqueeze(torch.cat(scores, 0), 1)
        labels = torch.unsqueeze(torch.cat(labels, 0), 1).type(scores.dtype)
        logger.info("Mean epoch MRR: {:.4f}".format(mrr(scores, labels)))

    return model


def save_embedding(model, graph, config):
    model.eval()
    batch_size = 200
    saving_dataset = MockHetGnnFileNodeLoader(
        graph=graph,
        batch_size=batch_size,
        model=model,
        sample_file=config["sample_file"],
    )

    data_loader = torch.utils.data.DataLoader(saving_dataset)
    with open(osp.join(config["data_dir"], "node_embedding.txt"), "w") as embed_file:
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


def evaluate_link_prediction_on_het_gnn(input_data_map, config):  # noqa: F811
    random.seed(0)
    train_num, test_num = a_a_collaborate_train_test(config, input_data_map)
    auc, f1 = evaluation.evaluate_link_prediction(config, train_num, test_num)
    logger.info("Link prediction AUC: {:.4f} F1 score:{:.4f}".format(auc, f1))

    assert auc > 0.73 and auc < 0.75
    assert f1 > 0.66 and f1 < 0.69


def evaluate_classification_on_het_gnn(config):  # noqa: F811
    random.seed(0)
    train_num, test_num = a_class_cluster_feature_setting(config)
    macroF1, microF1 = evaluation.evaluate_node_classification(
        train_num, test_num, config
    )

    logger.info(
        "Node classification microF1: {:.4f} macroF1:{:.4f}".format(microF1, macroF1)
    )
    assert macroF1 > 0.95
    assert microF1 > 0.95


if __name__ == "__main__":
    with prepare_local_test_files() as local_test_files:
        config = {
            "data_dir": local_test_files,  # path to save intermediate data for evaluation
            "neighbor_count": 10,  # number of neighbors to sample of each node
            "num_epochs": 2,  # number of epochs for training.
            "batch_size": 128,  # mini-batch size.
            "walk_length": 5,
            "dim": 128,  # dimension of embedding.
            "learning_rate": 0.01,
            "node_type": 0,  # node type to train/evaluate model
            "node_type_count": 3,
            "neighbor_count": 10,
            "feature_dim": 128,
            "feature_idx": 0,
            "sample_file": os.path.join(
                local_test_files, "academic", "a_node_list.txt"
            ),  # file contains node id and type to calculate embeddings
            "A_n": 28646,  # number of author nodes
            "P_n": 21044,  # number of paper nodes
            "V_n": 18,  # number of venue nodes
            "C_n": 4,  # number of node class labels
        }

        feat_data, adj_lists = load_data(config)
        het_input_data = init_het_input_data(config)
        graph = MockGraph(feat_data, adj_lists, config)
        model = train_academic_data(graph, config)
        save_embedding(model, graph, config)
        evaluate_link_prediction_on_het_gnn(het_input_data, config)
        evaluate_classification_on_het_gnn(config)
