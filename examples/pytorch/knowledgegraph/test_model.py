# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import pytest
import numpy as np
import numpy.testing as npt
import torch
import json
import os
import sys
from typing import Tuple
import urllib.request
import zipfile
import tempfile

from torch.utils.data import IterableDataset

from deepgnn.graph_engine import Graph, FeatureType, SamplingStrategy
from model import KGEModel  # type: ignore


class MockEdgeDataLoader(IterableDataset):
    def __init__(self, batch_size: int, model, graph: Graph):
        self.curr_node = 0
        self.batch_size = batch_size
        self.model = model
        self.graph = graph

    def __iter__(self):
        return self

    def __next__(self):
        self.curr_node += self.batch_size
        if self.curr_node >= 1025:
            raise StopIteration
        inputs = self.graph.sample_edges(self.batch_size, 0)
        context = self.model.query(self.graph, inputs)
        return context


class MockGraph(Graph):
    def __init__(self, edge_array, edge_fea, nodes_nei, nodes_res_nei):
        self.edge_array = np.array(edge_array, dtype=np.int64)
        self.edge_fea = edge_fea
        self.nodes_nei = nodes_nei
        self.nodes_res_nei = nodes_res_nei

    def sample_edges(
        self,
        size: int,
        edge_type: int,
        strategy: SamplingStrategy = SamplingStrategy.Random,
    ) -> np.ndarray:
        indexes = np.random.randint(0, len(self.edge_array), size)
        return self.edge_array[indexes]

    def edge_features(
        self, edges: np.ndarray, features: np.ndarray, feature_type: FeatureType
    ) -> np.ndarray:
        features = []

        for edge in edges:
            features.append(self.edge_fea[(edge[0], edge[1])])
        return np.array(features)

    def neighbors(
        self, nodes: np.ndarray, edge_type: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        neighbors = []
        neighbor_count = []
        for node_id in nodes:
            if edge_type == 0:
                neighbors.extend(self.nodes_nei[node_id])
                neighbor_count.append(len(self.nodes_nei[node_id]))
            else:
                neighbors.extend(self.nodes_res_nei[node_id])
                neighbor_count.append(len(self.nodes_res_nei[node_id]))
        return (
            np.array(neighbors, dtype=np.int64),
            [],
            [],
            np.array(neighbor_count, dtype=np.int32),
        )

    def __del__(self):
        os.remove("metadata.ini")


@pytest.fixture(scope="module")
def mock_graph():
    np.random.RandomState(seed=1)

    name = "FB15k.zip"
    working_dir = tempfile.TemporaryDirectory()
    zip_file = os.path.join(working_dir.name, name)
    urllib.request.urlretrieve(
        f"https://deepgraphpub.blob.core.windows.net/public/testdata/{name}", zip_file
    )
    with zipfile.ZipFile(zip_file, "r") as zip_ref:
        zip_ref.extractall(working_dir.name)

    entities_path = os.path.join(working_dir.name, "FB15k", "entities.dict")
    relations_path = os.path.join(working_dir.name, "FB15k", "relations.dict")
    train_path = os.path.join(working_dir.name, "FB15k", "train.txt")

    entities_dict = {}
    relations_dict = {}

    with open(entities_path, "r") as f:
        for line in f:
            line = line.rstrip()
            cols = line.split("\t")
            entities_dict[cols[1]] = int(cols[0])

    with open(relations_path, "r") as f:
        for line in f:
            line = line.rstrip()
            cols = line.split("\t")
            relations_dict[cols[1]] = int(cols[0])

    num_entities = len(entities_dict)
    num_relations = len(relations_dict)

    with open("metadata.ini", "w") as f:
        f.write("[DEFAULT]")
        f.write("\n")
        f.write("num_entities=" + str(num_entities))
        f.write("\n")
        f.write("num_relations=" + str(num_relations))
    edge_array = []
    nodes_nei = {}
    edge_fea = {}
    nodes_res_nei = {}
    start_ = 4

    count_ = {}
    with open(train_path, "r") as f:
        for line in f:
            line = line.rstrip()
            cols = line.split("\t")

            src_id = int(entities_dict[cols[0]])
            dst_id = int(entities_dict[cols[2]])
            rlt_id = int(relations_dict[cols[1]])

            edge_array.append([src_id, dst_id, rlt_id])
            edge_fea[src_id, dst_id] = [rlt_id, 0]
            if src_id not in nodes_nei:
                nodes_nei[src_id] = []
            nodes_nei[src_id].append(dst_id)

            if dst_id not in nodes_res_nei:
                nodes_res_nei[dst_id] = []
            nodes_res_nei[dst_id].append(src_id)
            if (src_id, rlt_id) not in count_:
                count_[(src_id, rlt_id)] = start_
            else:
                count_[(src_id, rlt_id)] += 1

            if (dst_id, -rlt_id - 1) not in count_:
                count_[(dst_id, -rlt_id - 1)] = start_
            else:
                count_[(dst_id, -rlt_id - 1)] += 1

        for edge in edge_array:
            subsampling_weight = (
                count_[(edge[0], edge[2])] + count_[(edge[1], -edge[2] - 1)]
            )
            edge_fea[(edge[0], edge[1])][1] = subsampling_weight

    yield MockGraph(edge_array, edge_fea, nodes_nei, nodes_res_nei)


def test_kg_rotatE_on_FB18k(mock_graph):
    np.random.seed(0)
    torch.manual_seed(0)

    num_negs = 10
    dim = 1000
    model_args = json.loads(
        '{"double_entity_embedding":1,'
        ' "adversarial_temperature":1.0,'
        '"regularization":0.0, "gamma":24.0,'
        ' "metadata_path":"metadata.ini",'
        ' "score_func":"RotatE"}'
    )

    kge_model = KGEModel(num_negs=num_negs, embed_dim=dim, model_args=model_args)
    optimizer = torch.optim.SGD(
        filter(lambda p: p.requires_grad, kge_model.parameters()), lr=0.7
    )

    trainloader = torch.utils.data.DataLoader(
        MockEdgeDataLoader(batch_size=256, model=kge_model, graph=mock_graph)
    )
    it = iter(trainloader)
    optimizer.zero_grad()
    loss, _, _ = kge_model(it.next())
    loss.backward()
    optimizer.step()

    npt.assert_allclose(loss.detach().numpy(), np.array([1.61642]), rtol=1e-5)


def test_kg_DistMult_on_FB18k(mock_graph):
    np.random.seed(0)
    torch.manual_seed(0)

    num_negs = 10
    dim = 1000
    model_args = json.loads(
        '{"double_entity_embedding":0,'
        ' "adversarial_temperature":1.0,'
        '"regularization":0.0, "gamma":24.0,'
        ' "metadata_path":"metadata.ini",'
        ' "score_func":"DistMult"}'
    )

    kge_model = KGEModel(num_negs=num_negs, embed_dim=dim, model_args=model_args)
    optimizer = torch.optim.SGD(
        filter(lambda p: p.requires_grad, kge_model.parameters()), lr=0.7
    )

    trainloader = torch.utils.data.DataLoader(
        MockEdgeDataLoader(batch_size=256, model=kge_model, graph=mock_graph)
    )
    it = iter(trainloader)
    optimizer.zero_grad()
    loss, _, _ = kge_model(it.next())
    loss.backward()
    optimizer.step()

    npt.assert_allclose(loss.detach().numpy(), np.array([0.693146]), rtol=1e-5)


def test_kg_ComplEx_on_FB18k(mock_graph):
    np.random.seed(0)
    torch.manual_seed(0)

    num_negs = 10
    dim = 1000
    model_args = json.loads(
        '{"double_entity_embedding":0,'
        ' "adversarial_temperature":1.0,'
        '"regularization":0.0, "gamma":24.0,'
        ' "metadata_path":"metadata.ini",'
        ' "score_func":"ComplEx"}'
    )

    kge_model = KGEModel(num_negs=num_negs, embed_dim=dim, model_args=model_args)
    optimizer = torch.optim.SGD(
        filter(lambda p: p.requires_grad, kge_model.parameters()), lr=0.7
    )

    trainloader = torch.utils.data.DataLoader(
        MockEdgeDataLoader(batch_size=256, model=kge_model, graph=mock_graph)
    )
    it = iter(trainloader)
    optimizer.zero_grad()
    loss, _, _ = kge_model(it.next())
    loss.backward()
    optimizer.step()

    npt.assert_allclose(loss.detach().numpy(), np.array([0.693146]), rtol=1e-5)


def test_kg_transE_on_FB18k(mock_graph):
    np.random.seed(0)
    torch.manual_seed(0)

    num_negs = 10
    dim = 1000
    model_args = json.loads(
        '{"double_entity_embedding":0,'
        ' "adversarial_temperature":1.0,'
        '"regularization":0.0, "gamma":24.0,'
        ' "metadata_path":"metadata.ini",'
        ' "score_func":"TransE"}'
    )

    kge_model = KGEModel(num_negs=num_negs, embed_dim=dim, model_args=model_args)
    optimizer = torch.optim.SGD(
        filter(lambda p: p.requires_grad, kge_model.parameters()), lr=0.7
    )

    trainloader = torch.utils.data.DataLoader(
        MockEdgeDataLoader(batch_size=256, model=kge_model, graph=mock_graph)
    )
    it = iter(trainloader)
    optimizer.zero_grad()
    loss, _, _ = kge_model(it.next())
    loss.backward()
    optimizer.step()

    npt.assert_allclose(loss.detach().numpy(), np.array([1.555402]), rtol=1e-5)


if __name__ == "__main__":
    sys.exit(
        pytest.main(
            [__file__, "--junitxml", os.environ["XML_OUTPUT_FILE"], *sys.argv[1:]]
        )
    )
