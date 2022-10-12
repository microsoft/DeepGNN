// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "src/cc/lib/distributed/client.h"
#include "src/cc/lib/distributed/server.h"
#include "src/cc/lib/graph/graph.h"
#include "src/cc/lib/graph/partition.h"
#include "src/cc/lib/graph/sampler.h"
#include "src/cc/lib/graph/xoroshiro.h"
#include "src/cc/tests/mocks.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <random>
#include <span>
#include <tuple>
#include <utility>
#include <vector>

#include "boost/random/uniform_int_distribution.hpp"
#include "gtest/gtest.h"

namespace
{
const size_t num_nodes = 100;
const size_t fv_size = 2;

// Helper to work with temporary folders: we run test with bazel,
// so it is ok to create local folders, because they run in a hermetic sandbox.
struct TempFolder
{
    std::filesystem::path path;
    bool last;

    explicit TempFolder(std::string name) : path(std::filesystem::temp_directory_path() / std::move(name))
    {
        if (std::filesystem::exists(path))
        {
            std::filesystem::remove_all(path);
        }
        if (!std::filesystem::create_directory(path))
        {
            throw std::logic_error("Failed to create path" + path.string());
        }

        last = true;
    }

    TempFolder(TempFolder &&other) : path(std::move(other.path))
    {
        other.last = false;
    }

    std::string string() const
    {
        return path.string();
    }

    ~TempFolder()
    {
        if (last)
        {
            std::filesystem::remove_all(path);
        }
    }
};
} // namespace

TEST(DistributedTest, NodeFeaturesSingleServer)
{
    const size_t num_nodes = 100;
    const size_t fv_size = 2;
    TestGraph::MemoryGraph m;
    for (size_t n = 0; n < num_nodes; n++)
    {
        std::vector<float> vals(fv_size);
        std::iota(std::begin(vals), std::end(vals), float(n));
        m.m_nodes.push_back(TestGraph::Node{
            .m_id = snark::NodeId(n), .m_type = 0, .m_weight = 1.0f, .m_float_features = {std::move(vals)}});
    }

    TempFolder path("NodeFeaturesSingleServer");
    auto partition = TestGraph::convert(path.path, "0_0", std::move(m), 1);

    snark::GRPCServer server(std::make_shared<snark::GraphEngineServiceImpl>(path.string(), std::vector<uint32_t>{0},
                                                                             snark::PartitionStorageType::memory, ""),
                             {}, "localhost:0", "", "", "");
    snark::GRPCClient c({server.InProcessChannel()}, 1, 1);

    std::vector<snark::NodeId> input_nodes = {0, 1, 2};
    std::vector<float> output(fv_size * input_nodes.size());
    std::vector<snark::FeatureMeta> features = {{snark::FeatureId(0), snark::FeatureSize(sizeof(float) * fv_size)}};
    c.GetNodeFeature(std::span(input_nodes), std::span(features),
                     std::span(reinterpret_cast<uint8_t *>(output.data()), sizeof(float) * output.size()));
    EXPECT_EQ(output, std::vector<float>({0, 1, 1, 2, 2, 3}));
}

TEST(DistributedTest, NodeStringFeaturesMultipleServers)
{
    const size_t num_nodes = 4;
    const size_t num_servers = 2;
    size_t start_node = 1;
    std::vector<std::unique_ptr<snark::GRPCServer>> servers;
    std::vector<std::shared_ptr<grpc::Channel>> channels;
    for (size_t server_index = 0; server_index < num_servers; ++server_index)
    {
        TestGraph::MemoryGraph m;
        for (size_t n = start_node; n < start_node + num_nodes; n++)
        {
            std::vector<float> vals_1(n);
            std::iota(std::begin(vals_1), std::end(vals_1), float(n));
            std::vector<float> vals_2 = {float(n), float(n - 1)};
            m.m_nodes.push_back(TestGraph::Node{.m_id = snark::NodeId(n),
                                                .m_type = 0,
                                                .m_weight = 1.0f,
                                                .m_float_features = {std::move(vals_1), std::move(vals_2)}});
        }
        start_node += num_nodes;

        TempFolder path("NodeStringFeaturesMultipleServers");
        auto partition = TestGraph::convert(path.path, "0_0", std::move(m), 1);

        servers.emplace_back(std::make_unique<snark::GRPCServer>(
            std::make_shared<snark::GraphEngineServiceImpl>(path.string(), std::vector<uint32_t>{0},
                                                            snark::PartitionStorageType::memory, ""),
            std::shared_ptr<snark::GraphSamplerServiceImpl>{}, "localhost:0", "", "", ""));
        channels.emplace_back(servers.back()->InProcessChannel());

        // Verify client correctly parses empty messages.
        servers.emplace_back(std::make_unique<snark::GRPCServer>(
            std::shared_ptr<snark::GraphEngineServiceImpl>{},
            std::make_shared<snark::GraphSamplerServiceImpl>(path.string(), std::set<size_t>{0}), "localhost:0", "", "",
            ""));
        channels.emplace_back(servers.back()->InProcessChannel());
    }

    snark::GRPCClient c(channels, 1, 1);

    std::vector<snark::NodeId> input_nodes = {2, 5, 0, 1};
    std::vector<snark::FeatureId> features = {1, 0};
    std::vector<uint8_t> values;
    std::vector<int64_t> dimensions(input_nodes.size() * features.size());
    c.GetNodeStringFeature(std::span(input_nodes), std::span(features), std::span(dimensions), values);
    std::span res(reinterpret_cast<float *>(values.data()), values.size() / 4);
    EXPECT_EQ(std::vector<float>(std::begin(res), std::end(res)),
              std::vector<float>({2, 1, 2, 3, 5, 4, 5, 6, 7, 8, 9, 1, 0, 1}));
    EXPECT_EQ(dimensions, std::vector<int64_t>({8, 8, 8, 20, 0, 0, 8, 4}));
}

TEST(DistributedTest, NodeSparseFeaturesMultipleServers)
{
    const size_t num_nodes = 4;
    const size_t num_servers = 1;
    size_t start_node = 1;
    std::vector<std::unique_ptr<snark::GRPCServer>> servers;
    std::vector<std::shared_ptr<grpc::Channel>> channels;
    for (size_t server_index = 0; server_index < num_servers; ++server_index)
    {
        TestGraph::MemoryGraph m;
        for (size_t n = start_node; n < start_node + num_nodes; n++)
        {
            std::vector<std::vector<float>> features;
            if (n == start_node)
            {
                std::vector<int32_t> f1_data = {{3, 3, 1, 0, 13, 0, 42, 0, 1}};
                auto start = reinterpret_cast<float *>(f1_data.data());
                features = {std::vector<float>(start, start + f1_data.size())};
            }
            m.m_nodes.push_back(TestGraph::Node{
                .m_id = snark::NodeId(n), .m_type = 0, .m_weight = 1.0f, .m_float_features = std::move(features)});
        }
        start_node += num_nodes;

        TempFolder path("NodeSparseFeaturesMultipleServers");
        auto partition = TestGraph::convert(path.path, "0_0", std::move(m), 1);

        servers.emplace_back(std::make_unique<snark::GRPCServer>(
            std::make_shared<snark::GraphEngineServiceImpl>(path.string(), std::vector<uint32_t>{0},
                                                            snark::PartitionStorageType::memory, ""),
            std::shared_ptr<snark::GraphSamplerServiceImpl>{}, "localhost:0", "", "", ""));
        channels.emplace_back(servers.back()->InProcessChannel());
    }
    snark::GRPCClient c(channels, 1, 1);

    std::vector<snark::NodeId> input_nodes = {2, 5, 0, 1};
    std::vector<snark::FeatureId> features = {1, 0};
    std::vector<std::vector<uint8_t>> values(features.size());
    std::vector<std::vector<int64_t>> indices(features.size());
    std::vector<int64_t> dimensions(features.size());
    c.GetNodeSparseFeature(std::span(input_nodes), std::span(features), std::span(dimensions), indices, values);
    std::span res(reinterpret_cast<int32_t *>(values[1].data()), values[1].size() / 4);
    EXPECT_EQ(std::vector<int32_t>(std::begin(res), std::end(res)), std::vector<int32_t>({1}));
    EXPECT_EQ(dimensions, std::vector<int64_t>({0, 3}));
}

TEST(DistributedTest, NodeSparseFeaturesSingleServerMissingFeatures)
{
    // indices - 17416, data - 1.0
    std::vector<int32_t> f5_data = {1, 1, 17416, 0, 1065353216};
    auto f5_start = reinterpret_cast<float *>(f5_data.data());
    // indices - 1, data - 1.0
    std::vector<int32_t> f6_data = {1, 1, 0, 0, 1065353216};
    auto f6_start = reinterpret_cast<float *>(f6_data.data());

    std::vector<std::vector<float>> input_features = {{},
                                                      {},
                                                      {},
                                                      {},
                                                      {},
                                                      std::vector<float>(f5_start, f5_start + f5_data.size()),
                                                      std::vector<float>(f6_start, f6_start + f6_data.size())};
    TestGraph::MemoryGraph m;
    m.m_nodes.push_back(TestGraph::Node{
        .m_id = snark::NodeId(13979298), .m_type = 0, .m_weight = 1.0f, .m_float_features = std::move(input_features)});
    TempFolder path("NodeSparseFeaturesSingleServerMissingFeatures");
    auto partition = TestGraph::convert(path.path, "0_0", std::move(m), 1);

    auto server = std::make_unique<snark::GRPCServer>(
        std::make_shared<snark::GraphEngineServiceImpl>(path.string(), std::vector<uint32_t>{0},
                                                        snark::PartitionStorageType::memory, ""),
        std::shared_ptr<snark::GraphSamplerServiceImpl>{}, "localhost:0", "", "", "");
    auto channel = server->InProcessChannel();

    snark::GRPCClient c({channel}, 1, 1);

    std::vector<snark::NodeId> nodes = {13979298};
    std::vector<snark::FeatureId> features = {6};

    std::vector<std::vector<uint8_t>> data(features.size());
    std::vector<std::vector<int64_t>> indices(features.size());
    std::vector<int64_t> dimensions = {-1};
    c.GetNodeSparseFeature(std::span(nodes), std::span(features), std::span(dimensions), indices, data);
    EXPECT_EQ(std::vector<int64_t>({0, 0}), indices[0]);
    EXPECT_EQ(std::vector<int64_t>({1}), dimensions);
    auto tmp = reinterpret_cast<float *>(data[0].data());
    EXPECT_EQ(std::vector<float>({1.0}), std::vector<float>(tmp, tmp + 1));

    features = {1, 6};

    data = {{}, {}};
    indices = {{}, {}};
    dimensions = {-1, -1};
    c.GetNodeSparseFeature(std::span(nodes), std::span(features), std::span(dimensions), indices, data);
    EXPECT_EQ(std::vector<int64_t>({}), indices[0]);
    EXPECT_EQ(std::vector<int64_t>({0, 0}), indices[1]);
    EXPECT_EQ(std::vector<int64_t>({0, 1}), dimensions);
    tmp = reinterpret_cast<float *>(data[1].data());
    EXPECT_EQ(std::vector<float>({1.0}), std::vector<float>(tmp, tmp + 1));

    features = {1, 2, 5, 6};
    data = {{}, {}, {}, {}};
    indices = {{}, {}, {}, {}};
    dimensions = {-1, -1, -1, -1};
    c.GetNodeSparseFeature(std::span(nodes), std::span(features), std::span(dimensions), indices, data);
    EXPECT_EQ(std::vector<int64_t>({}), indices[0]);
    EXPECT_EQ(std::vector<int64_t>({}), indices[1]);
    EXPECT_EQ(std::vector<int64_t>({0, 17416}), indices[2]);
    EXPECT_EQ(std::vector<int64_t>({0, 0}), indices[3]);
    EXPECT_EQ(std::vector<int64_t>({0, 0, 1, 1}), dimensions);
    EXPECT_EQ(0, data[0].size());
    EXPECT_EQ(0, data[1].size());
    tmp = reinterpret_cast<float *>(data[2].data());
    EXPECT_EQ(std::vector<float>({1.0}), std::vector<float>(tmp, tmp + 1));
    tmp = reinterpret_cast<float *>(data[3].data());
    EXPECT_EQ(std::vector<float>({1.0}), std::vector<float>(tmp, tmp + 1));

    features = {5, 6};
    data = {{}, {}};
    indices = {{}, {}};
    dimensions = {-1, -1};
    c.GetNodeSparseFeature(std::span(nodes), std::span(features), std::span(dimensions), indices, data);
    EXPECT_EQ(std::vector<int64_t>({0, 17416}), indices[0]);
    EXPECT_EQ(std::vector<int64_t>({0, 0}), indices[1]);
    EXPECT_EQ(std::vector<int64_t>({1, 1}), dimensions);
    EXPECT_EQ(sizeof(float), data[0].size());
    EXPECT_EQ(sizeof(float), data[1].size());
    tmp = reinterpret_cast<float *>(data[0].data());
    EXPECT_EQ(std::vector<float>({1.0}), std::vector<float>(tmp, tmp + 1));
    tmp = reinterpret_cast<float *>(data[1].data());
    EXPECT_EQ(std::vector<float>({1.0}), std::vector<float>(tmp, tmp + 1));
}

TEST(DistributedTest, NodeSparseFeaturesMultipleServersMissingFeatures)
{
    // std::vector<int32_t> f0_data = {1, 1, 1, 0, 1065353216};
    // auto f0_start = reinterpret_cast<float *>(f0_data.data());
    std::vector<int32_t> f1_data = {2, 2, 7, 0, 7, 0, 1065353216};
    auto f1_start = reinterpret_cast<float *>(f1_data.data());
    std::vector<int32_t> f2_data = {2, 2, 3, 0, 3, 0, 1065353216};
    auto f2_start = reinterpret_cast<float *>(f2_data.data());
    std::vector<int32_t> f3_data = {1, 1, 4, 0, 1065353216};
    auto f3_start = reinterpret_cast<float *>(f3_data.data());

    std::vector<std::unique_ptr<snark::GRPCServer>> servers;
    std::vector<std::shared_ptr<grpc::Channel>> channels;
    const size_t num_partitions = 4;
    std::vector<std::vector<std::vector<float>>> input_features = {
        std::vector<std::vector<float>>{{}, {}, std::vector<float>(f2_start, f2_start + f2_data.size())},
        std::vector<std::vector<float>>{{},
                                        {}, // std::vector<float>(f0_start, f0_start + f0_data.size()),
                                        std::vector<float>(f1_start, f1_start + f1_data.size())},
        std::vector<std::vector<float>>{{}, std::vector<float>(f3_start, f3_start + f3_data.size()), {}},
        std::vector<std::vector<float>>{},
    };
    for (size_t p = 0; p < num_partitions; ++p)
    {
        TestGraph::MemoryGraph m;
        m.m_nodes.push_back(TestGraph::Node{
            .m_id = snark::NodeId(p), .m_type = 0, .m_weight = 1.0f, .m_float_features = input_features[p]});

        TempFolder path("NodeSparseFeaturesMultipleServersMissingFeatures");
        auto partition = TestGraph::convert(path.path, "0_0", std::move(m), 1);

        servers.emplace_back(std::make_unique<snark::GRPCServer>(
            std::make_shared<snark::GraphEngineServiceImpl>(path.string(), std::vector<uint32_t>{0},
                                                            snark::PartitionStorageType::memory, ""),
            std::shared_ptr<snark::GraphSamplerServiceImpl>{}, "localhost:0", "", "", ""));
        channels.emplace_back(servers.back()->InProcessChannel());
    }

    snark::GRPCClient c({channels}, 1, 1);

    std::vector<snark::NodeId> nodes = {0, 1, 2, 3, 4};
    std::vector<snark::FeatureId> features = {1, 2};

    std::vector<std::vector<uint8_t>> data(features.size());
    std::vector<std::vector<int64_t>> indices(features.size());
    std::vector<int64_t> dimensions(features.size(), -1);
    c.GetNodeSparseFeature(std::span(nodes), std::span(features), std::span(dimensions), indices, data);
    // EXPECT_EQ(std::vector<int64_t>({1, 1, 2, 4}), indices[0]);
    EXPECT_EQ(std::vector<int64_t>({2, 4}), indices[0]);
    EXPECT_EQ(std::vector<int64_t>({0, 3, 3, 1, 7, 7}), indices[1]);
    EXPECT_EQ(std::vector<int64_t>({1, 2}), dimensions);
    auto tmp = reinterpret_cast<float *>(data[0].data());
    EXPECT_EQ(std::vector<float>({1.0}), std::vector<float>(tmp, tmp + 1));
    tmp = reinterpret_cast<float *>(data[1].data());
    EXPECT_EQ(std::vector<float>({1.0, 1.0}), std::vector<float>(tmp, tmp + 2));
}

TEST(DistributedTest, NodeSparseFeaturesServerMixWithEmptyGE)
{
    // indices - 1, data - 1.0
    std::vector<int32_t> sparse_data = {1, 1, 0, 0, 1065353216};
    auto sparse_start = reinterpret_cast<float *>(sparse_data.data());

    std::vector<std::vector<float>> input_features = {
        {}, std::vector<float>(sparse_start, sparse_start + sparse_data.size())};
    TestGraph::MemoryGraph m;
    m.m_nodes.push_back(TestGraph::Node{
        .m_id = snark::NodeId(42), .m_type = 0, .m_weight = 1.0f, .m_float_features = std::move(input_features)});
    TempFolder path("NodeSparseFeaturesSingleServerMissingFeatures");
    auto partition = TestGraph::convert(path.path, "0_0", std::move(m), 1);

    auto server = std::make_unique<snark::GRPCServer>(
        std::make_shared<snark::GraphEngineServiceImpl>(path.string(), std::vector<uint32_t>{0},
                                                        snark::PartitionStorageType::memory, ""),
        std::shared_ptr<snark::GraphSamplerServiceImpl>{}, "localhost:0", "", "", "");
    auto empty_server = std::make_unique<snark::GRPCServer>(
        std::shared_ptr<snark::GraphEngineServiceImpl>{},
        std::make_shared<snark::GraphSamplerServiceImpl>(path.string(), std::set<size_t>{0}), "localhost:0", "", "",
        "");

    snark::GRPCClient c({server->InProcessChannel(), empty_server->InProcessChannel()}, 1, 1);

    std::vector<snark::NodeId> nodes = {42};
    std::vector<snark::FeatureId> features = {1};

    std::vector<std::vector<uint8_t>> data(features.size());
    std::vector<std::vector<int64_t>> indices(features.size());
    std::vector<int64_t> dimensions = {-1};
    c.GetNodeSparseFeature(std::span(nodes), std::span(features), std::span(dimensions), indices, data);
    EXPECT_EQ(std::vector<int64_t>({0, 0}), indices[0]);
    EXPECT_EQ(std::vector<int64_t>({1}), dimensions);
    auto tmp = reinterpret_cast<float *>(data[0].data());
    EXPECT_EQ(std::vector<float>({1.0}), std::vector<float>(tmp, tmp + 1));
}

namespace
{
using TestChannels = std::vector<std::shared_ptr<grpc::Channel>>;
using TestServers = std::vector<std::unique_ptr<snark::GRPCServer>>;
std::pair<TestChannels, TestServers> MockServers(size_t num_partitions, std::string name, size_t num_node_types = 1)
{
    std::vector<std::unique_ptr<snark::GRPCServer>> servers;
    std::vector<std::shared_ptr<grpc::Channel>> channels;
    size_t curr_node = 0;
    for (size_t p = 0; p < num_partitions; ++p)
    {
        TestGraph::MemoryGraph m;
        for (size_t n = 0; n < num_nodes / num_partitions; n++, curr_node++)
        {
            std::vector<float> vals(fv_size);
            std::iota(std::begin(vals), std::end(vals), float(curr_node));
            m.m_nodes.push_back(TestGraph::Node{.m_id = snark::NodeId(curr_node),
                                                .m_type = int32_t(curr_node % num_node_types),
                                                .m_weight = 1.0f,
                                                .m_float_features = {std::move(vals)}});
        }

        TempFolder path(name);
        auto partition = TestGraph::convert(path.path, "0_0", std::move(m), num_node_types);

        servers.emplace_back(std::make_unique<snark::GRPCServer>(
            std::make_shared<snark::GraphEngineServiceImpl>(path.string(), std::vector<uint32_t>{0},
                                                            snark::PartitionStorageType::memory, ""),
            std::shared_ptr<snark::GraphSamplerServiceImpl>{}, "localhost:0", "", "", ""));
        channels.emplace_back(servers.back()->InProcessChannel());

        // Verify clients correctly process empty messages.
        servers.emplace_back(std::make_unique<snark::GRPCServer>(
            std::shared_ptr<snark::GraphEngineServiceImpl>{},
            std::make_shared<snark::GraphSamplerServiceImpl>(path.string(), std::set<size_t>{0}), "localhost:0", "", "",
            ""));
        channels.emplace_back(servers.back()->InProcessChannel());
    }

    return {std::move(channels), std::move(servers)};
}
} // namespace

TEST(DistributedTest, NodeFeaturesMultipleServers)
{
    auto mocks = MockServers(10, "NodeFeaturesMultipleServers");
    snark::GRPCClient c(std::move(mocks.first), 1, 1);

    std::vector<snark::NodeId> input_nodes = {0, 11, 22};
    std::vector<float> output(fv_size * input_nodes.size());
    std::vector<snark::FeatureMeta> features = {{snark::FeatureId(0), snark::FeatureSize(sizeof(float) * fv_size)}};
    c.GetNodeFeature(std::span(input_nodes), std::span(features),
                     std::span(reinterpret_cast<uint8_t *>(output.data()), sizeof(float) * output.size()));
    EXPECT_EQ(output, std::vector<float>({0, 1, 11, 12, 22, 23}));
}

TEST(DistributedTest, NodeTypeMultipleServers)
{
    auto mocks = MockServers(10, "NodeTypeMultipleServers", 3);
    snark::GRPCClient c(std::move(mocks.first), 1, 1);
    std::vector<snark::NodeId> input_nodes = {42, 0, 11, 22, 123};
    std::vector<snark::Type> types(5, -2);
    c.GetNodeType(std::span(input_nodes), std::span(types), -1);
    EXPECT_EQ(types, std::vector<snark::Type>({0, 0, 2, 1, -1}));
}

TEST(DistributedTest, NodeFeaturesMultipleServersMissingFeatureId)
{
    auto mocks = MockServers(10, "NodeFeaturesMultipleServersMissingFeatureId");
    snark::GRPCClient c(std::move(mocks.first), 1, 1);

    std::vector<snark::NodeId> input_nodes = {0, 11, 22};
    std::vector<float> output(fv_size * input_nodes.size(), -2);
    std::vector<snark::FeatureMeta> features = {{snark::FeatureId(12), snark::FeatureSize(sizeof(float) * fv_size)}};
    c.GetNodeFeature(std::span(input_nodes), std::span(features),
                     std::span(reinterpret_cast<uint8_t *>(output.data()), sizeof(float) * output.size()));
    EXPECT_EQ(output, std::vector<float>(fv_size * input_nodes.size(), 0));
}

TEST(DistributedTest, NodeFeaturesMultipleServersBackFillLargeRequestFeatureSize)
{
    auto mocks = MockServers(10, "NodeFeaturesMultipleServersBackFillLargeRequestFeatureSize");
    snark::GRPCClient c(std::move(mocks.first), 1, 1);

    std::vector<snark::NodeId> input_nodes = {0, 11, 22};
    std::vector<float> output(2 * fv_size * input_nodes.size(), -2);
    std::vector<snark::FeatureMeta> features = {{snark::FeatureId(0), snark::FeatureSize(2 * sizeof(float) * fv_size)}};
    c.GetNodeFeature(std::span(input_nodes), std::span(features),
                     std::span(reinterpret_cast<uint8_t *>(output.data()), sizeof(float) * output.size()));
    EXPECT_EQ(output, std::vector<float>({0, 1, 0, 0, 11, 12, 0, 0, 22, 23, 0, 0}));
}

std::pair<std::shared_ptr<snark::GRPCServer>, std::shared_ptr<snark::GRPCClient>> CreateSingleServerEnvironment(
    std::string name)
{
    TestGraph::MemoryGraph m;
    size_t curr_node = 0;
    const size_t num_nodes = 100;
    const size_t fv_size = 2;
    for (size_t n = 0; n < num_nodes; n++, curr_node++)
    {
        std::vector<float> vals(fv_size);
        std::iota(std::begin(vals), std::end(vals), float(n));
        m.m_nodes.push_back(TestGraph::Node{.m_id = snark::NodeId(curr_node),
                                            .m_type = 0,
                                            .m_weight = 1.0f,
                                            .m_neighbors = {TestGraph::NeighborRecord{curr_node + 1, 0, 1.0f},
                                                            TestGraph::NeighborRecord{curr_node + 2, 0, 2.0f},
                                                            TestGraph::NeighborRecord{curr_node + 3, 0, 2.0f},
                                                            TestGraph::NeighborRecord{curr_node + 4, 0, 1.0f}}});
    }

    TempFolder path(name);
    auto partition = TestGraph::convert(path.path, "0_0", std::move(m), 1);

    auto service = std::make_shared<snark::GraphEngineServiceImpl>(path.string(), std::vector<uint32_t>{0},
                                                                   snark::PartitionStorageType::memory, "");
    auto server = std::make_shared<snark::GRPCServer>(
        std::move(service), std::shared_ptr<snark::GraphSamplerServiceImpl>{}, "localhost:0", "", "", "");
    auto client = std::make_shared<snark::GRPCClient>(
        std::vector<std::shared_ptr<grpc::Channel>>{server->InProcessChannel()}, 1, 1);
    return {server, client};
}

TEST(DistributedTest, SampleNeighborsSingleServer)
{
    auto env = CreateSingleServerEnvironment("SampleNeighborsSingleServer");
    auto &client = *env.second;
    std::vector<snark::NodeId> input_nodes = {0, 1, 2};
    std::vector<snark::Type> input_types = {0};
    const size_t nb_count = 2;
    std::vector<snark::NodeId> output_nodes(nb_count * input_nodes.size());
    std::vector<float> output_weights(nb_count * input_nodes.size());
    std::vector<snark::Type> output_types(nb_count * input_nodes.size(), -1);
    client.WeightedSampleNeighbor(21, std::span(input_nodes), std::span(input_types), nb_count, std::span(output_nodes),
                                  std::span(output_types), std::span(output_weights), -1, 0.0f, -1);
    EXPECT_EQ(output_types, std::vector<snark::Type>(6, 0));
    EXPECT_EQ(output_nodes, std::vector<snark::NodeId>({3, 3, 3, 4, 5, 5}));
    EXPECT_EQ(output_weights, std::vector<float>({2, 2, 2, 2, 2, 2}));
}

TEST(DistributedTest, UniformSampleNeighborsSingleServer)
{
    auto env = CreateSingleServerEnvironment("UniformSampleNeighborsSingleServer");
    auto &client = *env.second;
    std::vector<snark::NodeId> input_nodes = {0, 1, 2};
    std::vector<snark::Type> input_types = {0};
    const size_t nb_count = 2;
    std::vector<snark::NodeId> output_nodes(nb_count * input_nodes.size());
    std::vector<snark::Type> output_types(nb_count * input_nodes.size(), -1);
    client.UniformSampleNeighbor(false, 21, std::span(input_nodes), std::span(input_types), nb_count,
                                 std::span(output_nodes), std::span(output_types), -1, -1);
    EXPECT_EQ(output_types, std::vector<snark::Type>({0, 0, 0, 0, 0, 0}));
    EXPECT_EQ(output_nodes, std::vector<snark::NodeId>({3, 3, 3, 5, 5, 6}));
}

TEST(DistributedTest, UniformSampleNeighborsWithoutReplacementSingleServer)
{
    auto env = CreateSingleServerEnvironment("UniformSampleNeighborsSingleServer");
    auto &client = *env.second;
    std::vector<snark::NodeId> input_nodes = {0, 1, 2};
    std::vector<snark::Type> input_types = {0};
    const size_t nb_count = 2;
    std::vector<snark::NodeId> output_nodes(nb_count * input_nodes.size());
    std::vector<snark::Type> output_types(nb_count * input_nodes.size(), -1);
    client.UniformSampleNeighbor(true, 21, std::span(input_nodes), std::span(input_types), nb_count,
                                 std::span(output_nodes), std::span(output_types), -1, -1);
    EXPECT_EQ(output_types, std::vector<snark::Type>({0, 0, 0, 0, 0, 0}));
    EXPECT_EQ(output_nodes, std::vector<snark::NodeId>({1, 3, 2, 4, 6, 3}));
}

using ServerList = std::vector<std::shared_ptr<snark::GRPCServer>>;
std::pair<ServerList, std::shared_ptr<snark::GRPCClient>> CreateMultiServerEnvironment(std::string name)
{
    const size_t num_servers = 10;
    ServerList servers;
    std::vector<std::shared_ptr<grpc::Channel>> channels;
    size_t curr_node = 0;
    for (size_t server = 0; server < num_servers; ++server)
    {
        TestGraph::MemoryGraph m;
        for (size_t n = 0; n < num_nodes / num_servers; n++, curr_node++)
        {
            std::vector<float> vals(fv_size);
            std::iota(std::begin(vals), std::end(vals), float(curr_node));
            m.m_nodes.push_back(TestGraph::Node{.m_id = snark::NodeId(curr_node),
                                                .m_type = 0,
                                                .m_weight = 1.0f,
                                                .m_neighbors = {TestGraph::NeighborRecord{curr_node + 1, 0, 1.0f},
                                                                TestGraph::NeighborRecord{curr_node + 2, 0, 2.0f},
                                                                TestGraph::NeighborRecord{curr_node + 3, 0, 1.0f},
                                                                TestGraph::NeighborRecord{curr_node + 4, 0, 2.0f}}});
        }

        TempFolder path(name);
        auto partition = TestGraph::convert(path.path, "0_0", std::move(m), 1);
        servers.emplace_back(std::make_shared<snark::GRPCServer>(
            std::make_shared<snark::GraphEngineServiceImpl>(path.string(), std::vector<uint32_t>{0},
                                                            snark::PartitionStorageType::memory, ""),
            std::shared_ptr<snark::GraphSamplerServiceImpl>{}, "localhost:0", "", "", ""));
        channels.emplace_back(servers.back()->InProcessChannel());
    }

    return std::make_pair(std::move(servers), std::make_shared<snark::GRPCClient>(std::move(channels), 1, 1));
}

TEST(DistributedTest, SampleNeighborsMultipleServers)
{
    auto environment = CreateMultiServerEnvironment("SampleNeighborsMultipleServers");
    auto &c = *environment.second;

    std::vector<snark::NodeId> input_nodes = {0, 55, 77};
    std::vector<snark::Type> input_types = {0};
    const size_t nb_count = 2;
    std::vector<snark::NodeId> output_nodes(nb_count * input_nodes.size());
    std::vector<float> output_weights(nb_count * input_nodes.size());
    std::vector<snark::Type> output_types(nb_count * input_nodes.size(), -1);
    c.WeightedSampleNeighbor(23, std::span(input_nodes), std::span(input_types), nb_count, std::span(output_nodes),
                             std::span(output_types), std::span(output_weights), -1, 0.0f, -1);
    EXPECT_EQ(output_types, std::vector<snark::Type>(6, 0));
    EXPECT_EQ(output_nodes, std::vector<snark::NodeId>({2, 2, 57, 56, 80, 81}));
    EXPECT_EQ(output_weights, std::vector<float>({2, 2, 2, 1, 1, 2}));
}

TEST(DistributedTest, SampleNeighborsMultipleServersMissingNeighbors)
{
    auto environment = CreateMultiServerEnvironment("SampleNeighborsMultipleServersMissingNeighbors");
    auto &c = *environment.second;

    std::vector<snark::NodeId> input_nodes = {0, 55, 77};
    std::vector<snark::Type> input_types = {1};
    const size_t nb_count = 2;
    std::vector<snark::NodeId> output_nodes(nb_count * input_nodes.size());
    std::vector<float> output_weights(nb_count * input_nodes.size());
    std::vector<snark::Type> output_types(nb_count * input_nodes.size(), -1);
    c.WeightedSampleNeighbor(23, std::span(input_nodes), std::span(input_types), nb_count, std::span(output_nodes),
                             std::span(output_types), std::span(output_weights), -1, 0.0f, -1);
    EXPECT_EQ(output_types, std::vector<snark::Type>(6, -1));
    EXPECT_EQ(output_nodes, std::vector<snark::NodeId>(6, -1));
    EXPECT_EQ(output_weights, std::vector<float>(6, 0));
}

TEST(DistributedTest, UniformSampleNeighborsMultipleServers)
{
    auto environment = CreateMultiServerEnvironment("UniformSampleNeighborsMultipleServers");
    auto &c = *environment.second;

    std::vector<snark::NodeId> input_nodes = {0, 55, 77};
    std::vector<snark::Type> input_types = {0};
    const size_t nb_count = 2;
    std::vector<snark::NodeId> output_nodes(nb_count * input_nodes.size());
    std::vector<snark::Type> output_types(nb_count * input_nodes.size(), -1);
    c.UniformSampleNeighbor(false, 23, std::span(input_nodes), std::span(input_types), nb_count,
                            std::span(output_nodes), std::span(output_types), -1, -1);
    EXPECT_EQ(output_types, std::vector<snark::Type>({0, 0, 0, 0, 0, 0}));
    EXPECT_EQ(output_nodes, std::vector<snark::NodeId>({2, 2, 57, 56, 80, 81}));
}

TEST(DistributedTest, UniformSampleNeighborsWithoutReplacementMultipleServers)
{
    auto environment = CreateMultiServerEnvironment("UniformSampleNeighborsWithoutReplacementMultipleServers");
    auto &c = *environment.second;

    std::vector<snark::NodeId> input_nodes = {0, 55, 77};
    std::vector<snark::Type> input_types = {0};
    const size_t nb_count = 2;
    std::vector<snark::NodeId> output_nodes(nb_count * input_nodes.size());
    std::vector<snark::Type> output_types(nb_count * input_nodes.size(), -1);
    c.UniformSampleNeighbor(true, 23, std::span(input_nodes), std::span(input_types), nb_count, std::span(output_nodes),
                            std::span(output_types), -1, -1);
    EXPECT_EQ(output_types, std::vector<snark::Type>({0, 0, 0, 0, 0, 0}));
    EXPECT_EQ(output_nodes, std::vector<snark::NodeId>({2, 4, 59, 57, 81, 79}));
}

TEST(DistributedTest, NeighborCountMultipleServers)
{
    const size_t num_servers = 2;
    ServerList servers;
    std::vector<std::shared_ptr<grpc::Channel>> channels;
    size_t curr_node = 0;
    for (size_t server = 0; server < num_servers; ++server)
    {
        TestGraph::MemoryGraph m;
        for (size_t n = 0; n < num_nodes / num_servers; n++, curr_node++)
        {
            std::vector<float> vals(fv_size);
            std::iota(std::begin(vals), std::end(vals), float(curr_node));
            m.m_nodes.push_back(TestGraph::Node{.m_id = snark::NodeId(curr_node),
                                                .m_type = 0,
                                                .m_weight = 1.0f,
                                                .m_neighbors = {TestGraph::NeighborRecord{curr_node + 1, 0, 1.0f},
                                                                TestGraph::NeighborRecord{curr_node + 2, 1, 2.0f},
                                                                TestGraph::NeighborRecord{curr_node + 3, 1, 1.0f},
                                                                TestGraph::NeighborRecord{curr_node + 4, 2, 2.0f}}});
        }

        TempFolder path("NeighborCountMultipleServers");
        auto partition = TestGraph::convert(path.path, "0_0", std::move(m), 1);
        servers.emplace_back(std::make_shared<snark::GRPCServer>(
            std::make_shared<snark::GraphEngineServiceImpl>(path.string(), std::vector<uint32_t>{0},
                                                            snark::PartitionStorageType::memory, ""),
            std::shared_ptr<snark::GraphSamplerServiceImpl>{}, "localhost:0", "", "", ""));
        channels.emplace_back(servers.back()->InProcessChannel());
    }

    snark::GRPCClient c(std::move(channels), 1, 1);
    std::vector<snark::NodeId> input_nodes = {0};
    std::vector<snark::Type> input_types = {0, 1};
    size_t size = input_nodes.size();
    std::vector<uint64_t> output_counts(size);
    std::fill_n(std::begin(output_counts), size, -1); // Fill with -1 to check update
    c.NeighborCount(std::span(input_nodes), std::span(input_types), std::span(output_counts));
    EXPECT_EQ(output_counts, std::vector<uint64_t>({3}));
}

TEST(DistributedTest, NeighborCountMismatchingOutputSize)
{
    const size_t num_servers = 2;
    ServerList servers;
    std::vector<std::shared_ptr<grpc::Channel>> channels;
    size_t curr_node = 0;
    for (size_t server = 0; server < num_servers; ++server)
    {
        TestGraph::MemoryGraph m;
        for (size_t n = 0; n < num_nodes / num_servers; n++, curr_node++)
        {
            std::vector<float> vals(fv_size);
            std::iota(std::begin(vals), std::end(vals), float(curr_node));
            m.m_nodes.push_back(TestGraph::Node{.m_id = snark::NodeId(curr_node),
                                                .m_type = 0,
                                                .m_weight = 1.0f,
                                                .m_neighbors = {TestGraph::NeighborRecord{curr_node + 1, 0, 1.0f},
                                                                TestGraph::NeighborRecord{curr_node + 2, 1, 2.0f},
                                                                TestGraph::NeighborRecord{curr_node + 3, 1, 1.0f},
                                                                TestGraph::NeighborRecord{curr_node + 4, 2, 2.0f}}});
        }

        TempFolder path("NeighborCountMismatchingOutputSize");
        auto partition = TestGraph::convert(path.path, "0_0", std::move(m), 1);
        servers.emplace_back(std::make_shared<snark::GRPCServer>(
            std::make_shared<snark::GraphEngineServiceImpl>(path.string(), std::vector<uint32_t>{0},
                                                            snark::PartitionStorageType::memory, ""),
            std::shared_ptr<snark::GraphSamplerServiceImpl>{}, "localhost:0", "", "", ""));
        channels.emplace_back(servers.back()->InProcessChannel());
    }

    snark::GRPCClient c(std::move(channels), 1, 1);
    std::vector<snark::NodeId> input_nodes = {0};
    std::vector<snark::Type> input_types = {0, 1};
    size_t size = input_nodes.size();

    // Make output counts larger than replies size to test mismatch
    std::vector<uint64_t> output_counts(size + 5);

    std::fill_n(std::begin(output_counts), size, -1); // Fill with -1 to check update
    c.NeighborCount(std::span(input_nodes), std::span(input_types), std::span(output_counts));
    EXPECT_EQ(output_counts, std::vector<uint64_t>({3, 0, 0, 0, 0, 0}));
}

TEST(DistributedTest, NeighborCountEmptyGraph)
{
    const size_t num_servers = 2;
    ServerList servers;
    std::vector<std::shared_ptr<grpc::Channel>> channels;
    size_t curr_node = 0;
    for (size_t server = 0; server < num_servers; ++server)
    {
        TestGraph::MemoryGraph m;
        for (size_t n = 0; n < num_nodes / num_servers; n++, curr_node++)
        {
            std::vector<float> vals(fv_size);
            std::iota(std::begin(vals), std::end(vals), float(curr_node));
            m.m_nodes.push_back(TestGraph::Node{.m_id = snark::NodeId(curr_node),
                                                .m_type = 0,
                                                .m_weight = 1.0f,
                                                .m_neighbors = {TestGraph::NeighborRecord{curr_node + 1, 0, 1.0f},
                                                                TestGraph::NeighborRecord{curr_node + 2, 1, 2.0f},
                                                                TestGraph::NeighborRecord{curr_node + 3, 1, 1.0f},
                                                                TestGraph::NeighborRecord{curr_node + 4, 2, 2.0f}}});
        }

        TempFolder path("NeighborCountMismatchingOutputSizeEmptyGraphEng");
        auto partition = TestGraph::convert(path.path, "0_0", std::move(m), 1);

        // EmptyGraphEngine as engine service
        servers.emplace_back(std::make_shared<snark::GRPCServer>(
            std::shared_ptr<snark::GraphEngineServiceImpl>{},
            std::make_shared<snark::GraphSamplerServiceImpl>(path.string(), std::set<size_t>{0}), "localhost:0", "", "",
            ""));
        channels.emplace_back(servers.back()->InProcessChannel());
    }

    snark::GRPCClient c(std::move(channels), 1, 1);
    std::vector<snark::NodeId> input_nodes = {0};
    std::vector<snark::Type> input_types = {};
    size_t size = input_nodes.size();

    // Make output counts larger than replies size to test mismatch
    std::vector<uint64_t> output_counts(size + 5);

    std::fill_n(std::begin(output_counts), size, -1); // Fill with -1 to check update
    c.NeighborCount(std::span(input_nodes), std::span(input_types), std::span(output_counts));
    EXPECT_EQ(output_counts, std::vector<uint64_t>({0, 0, 0, 0, 0, 0}));
}

TEST(DistributedTest, NeighborCountMultipleTypesMultipleServers)
{
    auto environment = CreateMultiServerEnvironment("NeighborCountMultipleTypesMultipleServers");
    auto &c = *environment.second;

    std::vector<snark::NodeId> input_nodes = {0, 55, 100};
    std::vector<snark::Type> input_types = {0};
    std::vector<uint64_t> output_counts(input_nodes.size());
    c.NeighborCount(std::span(input_nodes), std::span(input_types), std::span(output_counts));
    EXPECT_EQ(output_counts, std::vector<uint64_t>({4, 4, 0}));
}

TEST(DistributedTest, FullNeighborsMultipleTypesMultipleServers)
{
    const size_t num_servers = 2;
    ServerList servers;
    std::vector<std::shared_ptr<grpc::Channel>> channels;
    size_t curr_node = 0;
    for (size_t server = 0; server < num_servers; ++server)
    {
        TestGraph::MemoryGraph m;
        for (size_t n = 0; n < num_nodes / num_servers; n++, curr_node++)
        {
            std::vector<float> vals(fv_size);
            std::iota(std::begin(vals), std::end(vals), float(curr_node));
            m.m_nodes.push_back(TestGraph::Node{.m_id = snark::NodeId(curr_node),
                                                .m_type = 0,
                                                .m_weight = 1.0f,
                                                .m_neighbors = {TestGraph::NeighborRecord{curr_node + 1, 0, 1.0f},
                                                                TestGraph::NeighborRecord{curr_node + 2, 1, 2.0f},
                                                                TestGraph::NeighborRecord{curr_node + 3, 1, 1.0f},
                                                                TestGraph::NeighborRecord{curr_node + 4, 2, 2.0f}}});
        }

        TempFolder path("FullNeighborsMultipleTypesMultipleServers");
        auto partition = TestGraph::convert(path.path, "0_0", std::move(m), 1);
        servers.emplace_back(std::make_shared<snark::GRPCServer>(
            std::make_shared<snark::GraphEngineServiceImpl>(path.string(), std::vector<uint32_t>{0},
                                                            snark::PartitionStorageType::memory, ""),
            std::shared_ptr<snark::GraphSamplerServiceImpl>{}, "localhost:0", "", "", ""));
        channels.emplace_back(servers.back()->InProcessChannel());

        // Verify client correctly parses empty messages.
        servers.emplace_back(std::make_unique<snark::GRPCServer>(
            std::shared_ptr<snark::GraphEngineServiceImpl>{},
            std::make_shared<snark::GraphSamplerServiceImpl>(path.string(), std::set<size_t>{0}), "localhost:0", "", "",
            ""));
        channels.emplace_back(servers.back()->InProcessChannel());
    }

    snark::GRPCClient c(std::move(channels), 1, 1);
    std::vector<snark::NodeId> input_nodes = {0};
    std::vector<snark::Type> input_types = {0, 1};
    std::vector<snark::NodeId> output_nodes;
    std::vector<snark::Type> output_types;
    std::vector<float> output_weights;
    std::vector<uint64_t> output_counts(input_nodes.size());
    c.FullNeighbor(std::span(input_nodes), std::span(input_types), output_nodes, output_types, output_weights,
                   std::span(output_counts));
    EXPECT_EQ(output_types, std::vector<snark::Type>({0, 1, 1}));
    EXPECT_EQ(output_nodes, std::vector<snark::NodeId>({1, 2, 3}));
    EXPECT_EQ(output_weights, std::vector<float>({1, 2, 1}));
    EXPECT_EQ(output_counts, std::vector<uint64_t>({3}));
}

TEST(DistributedTest, FullNeighborsMultipleServers)
{
    auto environment = CreateMultiServerEnvironment("FullNeighborsMultipleServers");
    auto &c = *environment.second;

    std::vector<snark::NodeId> input_nodes = {0, 55};
    std::vector<snark::Type> input_types = {0};
    std::vector<snark::NodeId> output_nodes;
    std::vector<snark::Type> output_types;
    std::vector<float> output_weights;
    std::vector<uint64_t> output_counts(input_nodes.size());
    c.FullNeighbor(std::span(input_nodes), std::span(input_types), output_nodes, output_types, output_weights,
                   std::span(output_counts));
    EXPECT_EQ(output_types, std::vector<snark::Type>({0, 0, 0, 0, 0, 0, 0, 0}));
    EXPECT_EQ(output_nodes, std::vector<snark::NodeId>({1, 2, 3, 4, 56, 57, 58, 59}));
    EXPECT_EQ(output_weights, std::vector<float>({1, 2, 1, 2, 1, 2, 1, 2}));
    EXPECT_EQ(output_counts, std::vector<uint64_t>({4, 4}));
}

std::pair<ServerList, std::shared_ptr<snark::GRPCClient>> CreateMultiServerSplitFeaturesEnvironment(
    std::string name, std::vector<std::vector<float>> f0, std::vector<std::vector<float>> f1,
    std::vector<std::vector<float>> f2)
{
    const size_t num_servers = 2;
    ServerList servers;
    std::vector<std::shared_ptr<grpc::Channel>> channels;

    TestGraph::MemoryGraph m1;
    m1.m_nodes.push_back(TestGraph::Node{.m_id = 0, .m_type = 0, .m_weight = 1.0f, .m_float_features = f0});
    m1.m_nodes.push_back(TestGraph::Node{.m_id = 1, .m_type = 1, .m_weight = 1.0f, .m_float_features = f1});
    m1.m_nodes.push_back(TestGraph::Node{.m_id = 2, .m_type = -1, .m_weight = 1.0f});
    TestGraph::MemoryGraph m2;
    m2.m_nodes.push_back(TestGraph::Node{.m_id = 1, .m_type = -1});
    m2.m_nodes.push_back(TestGraph::Node{.m_id = 2, .m_type = 2, .m_float_features = f2});
    std::vector<TestGraph::MemoryGraph> test_graphs = {m1, m2};

    for (size_t server = 0; server < num_servers; ++server)
    {
        TempFolder path(name);
        auto partition = TestGraph::convert(path.path, "0_0", std::move(test_graphs[server]), 3);
        servers.emplace_back(std::make_shared<snark::GRPCServer>(
            std::make_shared<snark::GraphEngineServiceImpl>(path.string(), std::vector<uint32_t>{0},
                                                            snark::PartitionStorageType::memory, ""),
            std::shared_ptr<snark::GraphSamplerServiceImpl>{}, "localhost:0", "", "", ""));
        channels.emplace_back(servers.back()->InProcessChannel());
    }

    return std::make_pair(std::move(servers), std::make_shared<snark::GRPCClient>(std::move(channels), 1, 1));
}

TEST(DistributedTest, NodeTypesMultipleTypesNeighborsSpreadAcrossPartitions)
{
    std::vector<std::vector<float>> f0 = {std::vector<float>{1.0f, 2.0f, 3.0f}};
    std::vector<std::vector<float>> f1 = {std::vector<float>{4.0f, 5.0f, 6.0f}};
    std::vector<std::vector<float>> f2 = {std::vector<float>{7.0f, 8.0f, 9.0f}};

    auto environment =
        CreateMultiServerSplitFeaturesEnvironment("NodeTypesMultipleTypesNeighborsSpreadAcrossPartitions", f0, f1, f2);
    auto &c = *environment.second;

    std::vector<snark::NodeId> nodes = {0, 1, 2};
    std::vector<snark::Type> types(3, -3);

    c.GetNodeType(std::span(nodes), std::span(types), -2);
    EXPECT_EQ(std::vector<snark::Type>({0, 1, 2}), types);
}

TEST(DistributedTest, NodeFeaturesMultipleTypesNeighborsSpreadAcrossPartitions)
{
    std::vector<std::vector<float>> f0 = {std::vector<float>{1.0f, 2.0f, 3.0f}};
    std::vector<std::vector<float>> f1 = {std::vector<float>{4.0f, 5.0f, 6.0f}};
    std::vector<std::vector<float>> f2 = {std::vector<float>{7.0f, 8.0f, 9.0f}};

    auto environment = CreateMultiServerSplitFeaturesEnvironment(
        "NodeFeaturesMultipleTypesNeighborsSpreadAcrossPartitions", f0, f1, f2);
    auto &c = *environment.second;

    // 0 is a normal node
    // 1, 2 has a parity with type = -1
    // 3 is non existant
    std::vector<snark::NodeId> nodes = {0, 1, 2, 3};
    std::vector<uint8_t> output(4 * 3 * 4);
    std::vector<snark::FeatureMeta> features = {{0, 12}};

    c.GetNodeFeature(std::span(nodes), std::span(features), std::span(output));
    std::span res(reinterpret_cast<float *>(output.data()), output.size() / sizeof(float));
    EXPECT_EQ(std::vector<float>(std::begin(res), std::end(res)),
              std::vector<float>({1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 0, 0}));
}

TEST(DistributedTest, NodeStringFeaturesMultipleTypesNeighborsSpreadAcrossPartitions)
{
    std::vector<std::vector<float>> f0 = {std::vector<float>{1.0f, 2.0f, 3.0f}};
    std::vector<std::vector<float>> f1 = {std::vector<float>{4.0f, 5.0f, 6.0f}};
    std::vector<std::vector<float>> f2 = {std::vector<float>{7.0f, 8.0f, 9.0f}};

    auto environment = CreateMultiServerSplitFeaturesEnvironment(
        "NodeStringFeaturesMultipleTypesNeighborsSpreadAcrossPartitions", f0, f1, f2);
    auto &c = *environment.second;

    // 0 is a normal node
    // 1, 2 has a parity with type = -1
    // 3 is non existant
    std::vector<snark::NodeId> nodes = {0, 1, 2, 3};
    std::vector<uint8_t> output;
    std::vector<int64_t> dimensions(4);
    std::vector<snark::FeatureId> features = {0};
    c.GetNodeStringFeature(std::span(nodes), std::span(features), std::span(dimensions), output);
    std::span res(reinterpret_cast<float *>(output.data()), output.size() / sizeof(float));
    EXPECT_EQ(std::vector<float>(std::begin(res), std::end(res)), std::vector<float>({1, 2, 3, 4, 5, 6, 7, 8, 9}));
    EXPECT_EQ(dimensions, std::vector<int64_t>({12, 12, 12, 0}));
}

TEST(DistributedTest, NodeSparseFeaturesMultipleTypesNeighborsSpreadAcrossPartitions)
{
    // indices - 1, 14, 20, data - 1
    std::vector<int32_t> f0_data = {3, 3, 1, 0, 14, 0, 20, 0, 1};
    // indices - 1, 13, 42, data - 1
    std::vector<int32_t> f1_data = {3, 3, 1, 0, 13, 0, 42, 0, 1};
    // indices - [3, 8, 9], [4, 3, 2] data - [5, 42]
    std::vector<int32_t> f2_data = {6, 3, 3, 0, 8, 0, 9, 0, 4, 0, 3, 0, 2, 0, 5, 42};
    auto start = reinterpret_cast<float *>(f0_data.data());
    std::vector<std::vector<float>> f0 = {std::vector<float>(start, start + f0_data.size())};
    start = reinterpret_cast<float *>(f1_data.data());
    std::vector<std::vector<float>> f1 = {std::vector<float>(start, start + f1_data.size())};
    start = reinterpret_cast<float *>(f2_data.data());
    std::vector<std::vector<float>> f2 = {std::vector<float>(start, start + f2_data.size())};

    auto environment = CreateMultiServerSplitFeaturesEnvironment(
        "NodeSparseFeaturesMultipleTypesNeighborsSpreadAcrossPartitions", f0, f1, f2);
    auto &c = *environment.second;

    // 0 is a normal node
    // 1, 2 has a parity with type = -1
    // 3 is non existant
    std::vector<snark::NodeId> nodes = {0, 1, 2, 3};
    std::vector<snark::FeatureId> features = {0};
    std::vector<std::vector<uint8_t>> data(features.size());
    std::vector<std::vector<int64_t>> indices(features.size());
    std::vector<int64_t> dimensions = {-1};

    c.GetNodeSparseFeature(std::span(nodes), std::span(features), std::span(dimensions), indices, data);
    EXPECT_EQ(indices.size(), 1);
    EXPECT_EQ(data.size(), 1);
    EXPECT_EQ(std::vector<int64_t>({0, 1, 14, 20, 1, 1, 13, 42, 2, 3, 8, 9, 2, 4, 3, 2}), indices.front());
    auto tmp = reinterpret_cast<int32_t *>(data.front().data());
    EXPECT_EQ(std::vector<int32_t>({1, 1, 5, 42}), std::vector<int32_t>(tmp, tmp + 4));
    EXPECT_EQ(std::vector<int64_t>({3}), dimensions);
}

namespace
{

struct SamplerData
{
    std::unique_ptr<snark::GRPCClient> client;
    std::vector<std::unique_ptr<snark::GRPCServer>> servers;
    std::vector<TempFolder> dir_holders;

    SamplerData(size_t num_servers, size_t num_nodes_in_server, size_t num_edge_records_in_server,
                std::string path_prefix)
    {
        size_t curr_node = 0;
        size_t curr_src_node = 0;
        size_t curr_dst_node = 1;

        std::vector<std::shared_ptr<grpc::Channel>> channels;
        for (size_t p = 0; p < num_servers; ++p)
        {
            dir_holders.emplace_back(path_prefix + std::to_string(p));
            auto path = dir_holders.back().path;
            {
                auto alias = fopen((path / "node_0_0.alias").string().c_str(), "wb+");
                float prob = 0.5;
                for (size_t i = 0; i < num_nodes_in_server; ++i)
                {
                    EXPECT_EQ(1, fwrite(&curr_node, sizeof(curr_node), 1, alias));
                    ++curr_node;
                    EXPECT_EQ(1, fwrite(&curr_node, sizeof(curr_node), 1, alias));
                    EXPECT_EQ(1, fwrite(&prob, sizeof(prob), 1, alias));
                }
                EXPECT_EQ(0, fclose(alias));
            }
            {
                auto alias = fopen((path / "edge_0_0.alias").string().c_str(), "wb+");
                float prob = 0.5;
                for (size_t i = 0; i < num_edge_records_in_server; ++i)
                {
                    EXPECT_EQ(1, fwrite(&curr_src_node, sizeof(curr_src_node), 1, alias));
                    ++curr_src_node;
                    EXPECT_EQ(1, fwrite(&curr_dst_node, sizeof(curr_dst_node), 1, alias));
                    EXPECT_EQ(1, fwrite(&curr_src_node, sizeof(curr_src_node), 1, alias));
                    ++curr_dst_node;
                    EXPECT_EQ(1, fwrite(&curr_dst_node, sizeof(curr_dst_node), 1, alias));
                    EXPECT_EQ(1, fwrite(&prob, sizeof(prob), 1, alias));
                }
                EXPECT_EQ(0, fclose(alias));
            }
            {
                auto meta = fopen((path / "meta.txt").string().c_str(), "w+");
                fprintf(meta, "v%ld\n", snark::MINIMUM_SUPPORTED_VERSION);
                fprintf(meta, "%ld\n", num_nodes_in_server);
                fprintf(meta, "%ld\n", num_edge_records_in_server / 2);
                fprintf(meta, "1\n");                        // node_types_count
                fprintf(meta, "1\n");                        // edge_types_count
                fprintf(meta, "0\n");                        // node_features_count
                fprintf(meta, "0\n");                        // edge_features_count
                fprintf(meta, "1\n");                        // partition_count
                fprintf(meta, "0\n");                        // partition id
                fprintf(meta, "1\n");                        // partition node weight
                fprintf(meta, "1\n");                        // partition edge weight
                fprintf(meta, "%ld\n", num_nodes_in_server); // node count for type 0
                fprintf(meta, "%ld\n",
                        num_edge_records_in_server / 2); // edge count for type 0
                EXPECT_EQ(0, fclose(meta));
            }

            servers.emplace_back(std::make_unique<snark::GRPCServer>(
                std::shared_ptr<snark::GraphEngineServiceImpl>{},
                std::make_shared<snark::GraphSamplerServiceImpl>(path.string(), std::set<size_t>{0}), "localhost:0", "",
                "", ""));
            channels.emplace_back(servers.back()->InProcessChannel());
        }

        client = std::make_unique<snark::GRPCClient>(std::move(channels), 1, 1);
    }
};
} // namespace

TEST(DistributedTest, TestNodeSamplerSimple)
{
    SamplerData s(5, 4, 0, "DistributedTestNodeSamplerSimple");

    std::vector<snark::NodeId> output_nodes(7);
    std::vector<snark::Type> output_types(7, -1);
    std::vector<snark::Type> input_type = {0};
    auto sampler_id = s.client->CreateSampler(
        false, snark::CreateSamplerRequest_Category::CreateSamplerRequest_Category_WEIGHTED, std::span(input_type));
    EXPECT_EQ(sampler_id, 0); // There should be only one sampler initially.
    s.client->SampleNodes(3, sampler_id, std::span(output_nodes), std::span(output_types));
    EXPECT_EQ(output_nodes, std::vector<snark::NodeId>({0, 4, 4, 9, 12, 13, 15}));
    EXPECT_EQ(output_types, std::vector<snark::Type>({0, 0, 0, 0, 0, 0, 0}));
}

TEST(DistributedTest, TestNodeSamplerStatisticalProperties)
{
    const size_t num_servers = 3;
    const size_t num_nodes_per_server = 8;
    SamplerData s(num_servers, num_nodes_per_server, 0, "DistributedTestNodeSamplerStatisticalProperties");
    std::vector<size_t> counts(num_servers * num_nodes_per_server + 1);
    std::vector<snark::NodeId> output_nodes(5);
    std::vector<snark::Type> output_types(5, -1);
    std::vector<snark::Type> input_type = {0};
    auto sampler_id = s.client->CreateSampler(
        false, snark::CreateSamplerRequest_Category::CreateSamplerRequest_Category_WEIGHTED, std::span(input_type));
    EXPECT_EQ(sampler_id, 0); // There should be only one sampler initially.
    snark::Xoroshiro128PlusGenerator engine(42);
    engine.seed(42);
    boost::random::uniform_int_distribution<int64_t> seed;
    for (size_t i = 0; i < 1000; ++i)
    {
        s.client->SampleNodes(seed(engine), sampler_id, std::span(output_nodes), std::span(output_types));
        EXPECT_EQ(output_types, std::vector<snark::Type>({0, 0, 0, 0, 0}));
        for (auto node_id : output_nodes)
        {
            counts[node_id]++;
        }
    }

    // First and last node appear in alias tables only once
    // Their expected average counts are: 5*1000/(24 * 2) ~ 104.
    // Remaining nodes: 5*1000/24 ~ 208
    EXPECT_EQ(counts, std::vector<size_t>({93,  198, 198, 192, 222, 207, 201, 199, 213, 216, 213, 201, 226,
                                           197, 211, 211, 218, 208, 214, 194, 222, 237, 214, 189, 106}));
}

TEST(DistributedTest, TestEdgeSamplerSimple)
{
    SamplerData s(5, 0, 4, "DistributedTestEdgeSamplerSimple");

    std::vector<snark::NodeId> output_src_nodes(4, -1);
    std::vector<snark::NodeId> output_dst_nodes(4, -1);
    std::vector<snark::Type> output_types(4, -1);
    std::vector<snark::Type> input_type = {0};
    auto sampler_id = s.client->CreateSampler(
        true, snark::CreateSamplerRequest_Category::CreateSamplerRequest_Category_WEIGHTED, std::span(input_type));
    EXPECT_EQ(sampler_id, 0); // There should be only one sampler initially.
    s.client->SampleEdges(13, sampler_id, std::span(output_src_nodes), std::span(output_types),
                          std::span(output_dst_nodes));
    EXPECT_EQ(output_src_nodes, std::vector<snark::NodeId>({6, 7, 8, 16}));
    EXPECT_EQ(output_dst_nodes, std::vector<snark::NodeId>({7, 8, 9, 17}));
    EXPECT_EQ(output_types, std::vector<snark::Type>({0, 0, 0, 0}));
}

TEST(DistributedTest, TestEdgeSamplerStatisticalProperties)
{
    const size_t batch_size = 5;
    const size_t num_servers = 3;
    const size_t num_edges_per_server = 4;
    SamplerData s(num_servers, size_t{}, num_edges_per_server, "DistributedTestEdgeSamplerStatisticalProperties");
    std::vector<size_t> counts(num_servers * num_edges_per_server + 2);

    std::vector<snark::NodeId> output_src_nodes(batch_size);
    std::vector<snark::NodeId> output_dst_nodes(batch_size);
    std::vector<snark::Type> output_types(batch_size, -1);
    std::vector<snark::Type> input_type = {0};
    auto sampler_id = s.client->CreateSampler(
        true, snark::CreateSamplerRequest_Category::CreateSamplerRequest_Category_WEIGHTED, std::span(input_type));
    EXPECT_EQ(sampler_id, 0); // There should be only one sampler initially.
    snark::Xoroshiro128PlusGenerator engine(42);
    boost::random::uniform_int_distribution<int64_t> seed;
    for (size_t i = 0; i < 1000; ++i)
    {
        s.client->SampleEdges(seed(engine), sampler_id, std::span(output_src_nodes), std::span(output_types),
                              std::span(output_dst_nodes));
        EXPECT_EQ(output_types, std::vector<snark::Type>({0, 0, 0, 0, 0}));
        for (auto node_id : output_src_nodes)
        {
            counts[node_id]++;
        }
        for (auto node_id : output_dst_nodes)
        {
            counts[node_id]++;
        }
    }

    // First and last node appear in alias tables only once
    // Second and second to last appear only 3 times, remaining - 4 times.
    // Their expected average counts are: 5000/24, 15000/24, 20000/24
    EXPECT_EQ(counts, std::vector<size_t>({195, 583, 810, 832, 809, 835, 862, 847, 849, 828, 847, 879, 628, 196}));
}

TEST(DistributedTest, TestFetchNodeFeaturesFromOnlySamplerServer)
{
    const size_t fv_size = 2;
    const size_t num_servers = 10;
    const size_t num_nodes_per_server = 5;
    SamplerData s(num_servers, num_nodes_per_server, 0, "DistributedTestFetchNodeFeaturesFromOnlySamplerServer");
    std::vector<snark::NodeId> input_nodes = {0, 1, 2};
    std::vector<float> output(fv_size * input_nodes.size(), -2);
    std::vector<snark::FeatureMeta> features = {{snark::FeatureId(0), snark::FeatureSize(fv_size * 4)}};
    s.client->GetNodeFeature(std::span(input_nodes), std::span(features),
                             std::span(reinterpret_cast<uint8_t *>(output.data()), sizeof(float) * output.size()));
    EXPECT_EQ(output, std::vector<float>(fv_size * input_nodes.size(), 0));
}

TEST(DistributedTest, TestSamplerFromGEOnlyServer)
{
    TestGraph::MemoryGraph m;
    const size_t num_nodes = 100;
    const size_t fv_size = 2;
    for (size_t n = 0; n < num_nodes; n++)
    {
        std::vector<float> vals(fv_size);
        std::iota(std::begin(vals), std::end(vals), float(n));
        m.m_nodes.push_back(TestGraph::Node{
            .m_id = snark::NodeId(n), .m_type = 0, .m_weight = 1.0f, .m_float_features = {std::move(vals)}});
    }

    TempFolder path("TestSamplerFromGEOnlyServer");
    auto partition = TestGraph::convert(path.path, "0_0", std::move(m), 1);
    snark::GRPCServer server(std::make_shared<snark::GraphEngineServiceImpl>(path.string(), std::vector<uint32_t>{0},
                                                                             snark::PartitionStorageType::memory, ""),
                             {}, "localhost:0", "", "", "");
    snark::GRPCClient client({server.InProcessChannel()}, 1, 1);
    std::vector<snark::NodeId> output_nodes(3, -2);
    std::vector<snark::Type> output_types(3, -1);

    std::vector<snark::Type> input_type = {0};
    auto sampler_id = client.CreateSampler(
        false, snark::CreateSamplerRequest_Category::CreateSamplerRequest_Category_WEIGHTED, std::span(input_type));
    EXPECT_EQ(sampler_id, 0);
    client.SampleNodes(2, sampler_id, std::span(output_nodes), std::span(output_types));
    EXPECT_EQ(output_nodes, std::vector<snark::NodeId>(3, -2));
    EXPECT_EQ(output_types, std::vector<snark::Type>(3, -1));
}
