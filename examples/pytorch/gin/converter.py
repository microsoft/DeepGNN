from cgi import test
import os
from platform import node
import torch
import json
import argparse
import numpy as np
import networkx as nx
from pathlib import Path

from sklearn.model_selection import StratifiedKFold

import deepgnn.graph_engine.snark.convert as convert
from deepgnn.graph_engine.snark.decoders import JsonDecoder
from deepgnn.graph_engine.snark.decoders import DecoderType
import deepgnn.graph_engine.snark.client as client

class S2VGraph(object):
    def __init__(self, g, label, node_tags=None, node_features=None):
        '''
            g: a networkx graph
            label: an integer graph label
            node_tags: a list of integer node tags
            node_features: a torch float tensor, one-hot representation of the tag that is used as input to neural nets
            edge_mat: a torch long tensor, contain edge list, will be used to create torch sparse tensor
            neighbors: list of neighbors (without self-loop)
        '''
        self.label = label
        self.g = g
        self.node_tags = node_tags
        self.neighbors = []
        self.node_features = 0
        self.edge_mat = 0
        self.max_neighbor = 0

def convert_data(dataset, degree_as_tag):
    """"Convert dataset .txt file to networkx graph."""

    print('loading data')
    g_list = []
    label_dict = {}
    feat_dict = {}

    with open('dataset/%s/%s.txt' % (dataset, dataset), 'r') as f:
        n_g = int(f.readline().strip())
        for i in range(n_g):
            row = f.readline().strip().split()
            n, l = [int(w) for w in row]
            if not l in label_dict:
                mapped = len(label_dict)
                label_dict[l] = mapped
            g = nx.Graph()
            node_tags = []
            node_features = []
            n_edges = 0
            for j in range(n):
                g.add_node(j)
                row = f.readline().strip().split()
                tmp = int(row[1]) + 2
                if tmp == len(row):
                    # no node attributes
                    row = [int(w) for w in row]
                    attr = None
                else:
                    row, attr = [int(w) for w in row[:tmp]], np.array([float(w) for w in row[tmp:]])
                if not row[0] in feat_dict:
                    mapped = len(feat_dict)
                    feat_dict[row[0]] = mapped
                node_tags.append(feat_dict[row[0]])

                if tmp > len(row):
                    node_features.append(attr)

                n_edges += row[1]
                for k in range(2, len(row)):
                    g.add_edge(j, row[k])

            if node_features != []:
                node_features = np.stack(node_features)
                node_feature_flag = True
            else:
                node_features = None
                node_feature_flag = False

            assert len(g) == n
            # print(node_tags)
            g_list.append(S2VGraph(g, l, node_tags))

    #add labels and edge_mat       
    for g in g_list:
        g.neighbors = [[] for i in range(len(g.g))]
        for i, j in g.g.edges():
            g.neighbors[i].append(j)
            g.neighbors[j].append(i)
        degree_list = []
        for i in range(len(g.g)):
            g.neighbors[i] = g.neighbors[i]
            degree_list.append(len(g.neighbors[i]))
        g.max_neighbor = max(degree_list)

        g.label = label_dict[g.label]

        edges = [list(pair) for pair in g.g.edges()]
        edges.extend([[i, j] for j, i in edges])

        deg_list = list(dict(g.g.degree(range(len(g.g)))).values())
        g.edge_mat = torch.LongTensor(edges).transpose(0,1)

    if degree_as_tag:
        for g in g_list:
            g.node_tags = list(dict(g.g.degree).values())

    #Extracting unique tag labels   
    tagset = set([])
    for g in g_list:
        tagset = tagset.union(set(g.node_tags))

    tagset = list(tagset)
    tag2index = {tagset[i]:i for i in range(len(tagset))}

    for g in g_list:
        g.node_features = torch.zeros(len(g.node_tags), len(tagset))
        g.node_features[range(len(g.node_tags)), [tag2index[tag] for tag in g.node_tags]] = 1


    print('# classes: %d' % len(label_dict))
    print('# maximum node tag: %d' % len(tagset))
    print("# data: %d" % len(g_list))

    return g_list, len(label_dict)

def build_json(g_list, train_idx, test_idx):
    """"Generate graph.json file from networkx graph."""
    

    nodes = []
    data = ""
    idx = 0
    for g in g_list:

        label = g.label
        
        graph_ids = {}
        for node_id in g.g:
            graph_ids[node_id] = idx
            idx += 1
        # Fetch networkx graph from SV2Graph object
        # 3, 7, 9
        # 1, 2, 3
        for node_id in g.g:
            # Fetch and set weights for neighbors
            nbs = {}
            for nb in nx.neighbors(g.g, node_id):
                nbs[nb] = 1.0

            feats = g.node_features[node_id].tolist()
            # print(len(feats))

            node = {
                "node_weight": 1.0,
                "node_id": graph_ids[node_id],
                "node_type": 0,
                "float_feature": {"0": feats, "1": label},
                "edge": [{
                    "src_id": graph_ids[node_id],
                    "dst_id": graph_ids[nb],
                    "edge_type": 0,
                    "weight": 1.0
                }
                for nb in nx.neighbors(g.g, node_id)],
            }

            data += json.dumps(node) + "\n"
            nodes.append(node_id)

    return data


def seperate(graphs, fold_idx, seed):
    assert 0 <= fold_idx and fold_idx < 10, "fold_idx must be from 0 to 9."

    skf = StratifiedKFold(n_splits=10, shuffle = True, random_state = seed)

    labels = []
    nodes = []
    for graph in graphs:
        for node_id in graph.g:
            if node_id not in nodes:
                nodes.append(node_id)
                label = graph.node_features[node_id].tolist()[1]
                labels.append(label)

    idx_list = []
    # print(len(labels))
    # print(len(nodes))
    # nodes = list(nodes)

    for idx in skf.split(np.zeros(len(labels)), labels):
        idx_list.append(idx)

    train_idx, test_idx = idx_list[fold_idx]

    print(train_idx)
    print(test_idx)

    return train_idx, test_idx

def _main():

    parser = argparse.ArgumentParser(description='Loading graphs to json for training/evaluation.')
    parser.add_argument('--dataset', type=str, default="PROTEINS")
    args = parser.parse_args()

    # Build networkx graph from .txt file
    g_list = convert_data(args.dataset, True)[0]

    train_idx, test_idx = seperate(g_list, 0, 123)

    # Build json data from networkx and node features
    data = build_json(g_list, train_idx, test_idx)

    # print(data)
    data_dir = '/tmp/' + str.lower(args.dataset)

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    raw_file = "/tmp/" + str.lower(args.dataset) + "/graph.json"
    with open(raw_file, "w+") as f:
        f.write(data)
    
    test_file = "/tmp/"+ str.lower(args.dataset) +"/test.nodes"
    with open(test_file, "w+") as f:
        for idx in test_idx:
            f.write(str(idx) + '\n')

    # Build extra binaries
    convert.MultiWorkersConverter(
        graph_path=raw_file,
        partition_count=1,
        output_dir="/tmp/" + str.lower(args.dataset),
        decoder=JsonDecoder,
    ).convert()


if __name__ == "__main__":
    _main()