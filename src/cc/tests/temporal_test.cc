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
#include <filesystem>
#include <span>
#include <type_traits>
#include <vector>

#include "boost/random/uniform_int_distribution.hpp"
#include "gtest/gtest.h"

class TemporalTest : public ::testing::Test
{
  protected:
    std::filesystem::path m_path;
    std::unique_ptr<snark::Graph> m_single_partition_graph;
    std::unique_ptr<snark::Graph> m_multi_partition_graph;
    std::vector<std::shared_ptr<snark::GRPCServer>> m_servers;
    std::shared_ptr<snark::GRPCClient> m_distributed_graph;

    std::unique_ptr<snark::Graph> m_live_memory_graph;
    std::vector<std::shared_ptr<snark::GRPCServer>> m_live_memory_servers;
    std::shared_ptr<snark::GRPCClient> m_live_distributed_graph;

    void SetUp() override
    {
        m_path = std::filesystem::temp_directory_path() / "TemporalTests";
        std::filesystem::create_directory(m_path);
        {
            TestGraph::MemoryGraph m1;
            m1.m_nodes.push_back(
                TestGraph::Node{.m_id = 0,
                                .m_type = 0,
                                .m_weight = 1.0f,
                                .m_neighbors{std::vector<TestGraph::NeighborRecord>{{1, 0, 1.0f}, {2, 0, 2.0f}}}});

            m1.m_nodes.push_back(TestGraph::Node{
                .m_id = 1,
                .m_type = 1,
                .m_weight = 1.0f,
                .m_neighbors{std::vector<TestGraph::NeighborRecord>{{3, 0, 1.0f}, {4, 0, 1.0f}, {5, 1, 7.0f}}}});

            m1.m_watermark = 1;
            using ts_pair = std::pair<snark::Timestamp, snark::Timestamp>;
            m1.m_edge_timestamps.emplace_back(ts_pair{0, 1});
            m1.m_edge_timestamps.emplace_back(ts_pair{0, 1});
            m1.m_edge_timestamps.emplace_back(ts_pair{0, 1});
            m1.m_edge_timestamps.emplace_back(ts_pair{1, 2});
            m1.m_edge_timestamps.emplace_back(ts_pair{2, 3});

            // Initialize graph
            auto path = m_path / "single_partition";
            std::filesystem::create_directory(path);
            TestGraph::convert(path, "0_0", std::move(m1), 2);
            snark::Metadata metadata(path.string());
            m_single_partition_graph =
                std::make_unique<snark::Graph>(std::move(metadata), std::vector<std::string>{path.string()},
                                               std::vector<uint32_t>{0}, snark::PartitionStorageType::memory);
        }
        {
            auto path = m_path / "multi_partition";
            std::filesystem::create_directory(path);
            TestGraph::MemoryGraph m1;
            m1.m_nodes.push_back(
                TestGraph::Node{.m_id = 0,
                                .m_type = 0,
                                .m_weight = 1.0f,
                                .m_neighbors{std::vector<TestGraph::NeighborRecord>{{1, 0, 1.0f}, {2, 0, 2.0f}}}});

            m1.m_nodes.push_back(TestGraph::Node{
                .m_id = 1,
                .m_type = 1,
                .m_weight = 1.0f,
                .m_neighbors{std::vector<TestGraph::NeighborRecord>{{3, 0, 1.0f}, {4, 0, 1.0f}, {5, 1, 1.0f}}}});

            m1.m_watermark = 2;
            using ts_pair = std::pair<snark::Timestamp, snark::Timestamp>;
            m1.m_edge_timestamps.emplace_back(ts_pair{0, 1});
            m1.m_edge_timestamps.emplace_back(ts_pair{0, 1});
            m1.m_edge_timestamps.emplace_back(ts_pair{0, 1});
            m1.m_edge_timestamps.emplace_back(ts_pair{1, -1});
            m1.m_edge_timestamps.emplace_back(ts_pair{1, -1});
            TestGraph::MemoryGraph m2;
            m2.m_nodes.push_back(
                TestGraph::Node{.m_id = 1,
                                .m_type = 1,
                                .m_neighbors{std::vector<TestGraph::NeighborRecord>{{6, 1, 1.5f}, {7, 1, 3.0f}}}});

            m2.m_watermark = 3;
            m2.m_edge_timestamps.emplace_back(ts_pair{0, 1});
            m2.m_edge_timestamps.emplace_back(ts_pair{2, 3});

            // Initialize Graph
            TestGraph::convert(path, "0_0", std::move(m1), 2);
            TestGraph::convert(path, "1_0", std::move(m2), 2);
            snark::Metadata metadata(path.string());
            m_multi_partition_graph = std::make_unique<snark::Graph>(
                std::move(metadata), std::vector<std::string>{path.string(), path.string()},
                std::vector<uint32_t>{0, 1}, snark::PartitionStorageType::memory);
        }
        {
            // Reuse data from multiparittion in memory graph.
            std::vector<std::shared_ptr<grpc::Channel>> channels;
            auto path = m_path / "multi_partition";
            for (uint32_t server_index = 0; server_index < 2; ++server_index)
            {
                auto service = std::make_shared<snark::GraphEngineServiceImpl>(
                    snark::Metadata(path.string()), std::vector<std::string>{path.string()},
                    std::vector<uint32_t>{server_index}, snark::PartitionStorageType::memory);
                m_servers.push_back(std::make_shared<snark::GRPCServer>(
                    std::move(service), std::shared_ptr<snark::GraphSamplerServiceImpl>{}, std::string{"localhost:0"},
                    std::string{}, std::string{}, std::string{}));
                channels.emplace_back(m_servers.back()->InProcessChannel());
            }

            m_distributed_graph = std::make_shared<snark::GRPCClient>(std::move(channels), 1, 1);
        }

        {
            // Graph with 2 nodes spread across 2 partitions.
            // Node with id 0 has 2 neighbors of type 0 and 3 neighbors of type 1. It belongs to partition 0.
            // Node with id 1 has 4 neighbors of type 0 and 5 neighbors of type 1. 2 edges of type 0 belong to partition
            // 0, rest belong to partition 1.
            auto path = m_path / "live_edges";
            std::filesystem::create_directory(path);
            TestGraph::MemoryGraph m1;
            m1.m_nodes.push_back(
                TestGraph::Node{.m_id = 0,
                                .m_type = 0,
                                .m_weight = 1.0f,
                                .m_neighbors{std::vector<TestGraph::NeighborRecord>{
                                    {1, 0, 1.0f}, {2, 0, 2.0f}, {3, 1, 3.0f}, {4, 1, 4.0f}, {5, 1, 5.0f}}}});
            m1.m_nodes.push_back(
                TestGraph::Node{.m_id = 1,
                                .m_type = 1,
                                .m_neighbors{std::vector<TestGraph::NeighborRecord>{{6, 0, 1.5f}, {7, 0, 3.0f}}}});

            m1.m_watermark = 2;
            using ts_pair = std::pair<snark::Timestamp, snark::Timestamp>;
            for (uint32_t i = 0; i < 7; ++i)
            {
                m1.m_edge_timestamps.emplace_back(ts_pair{0, -1});
            }

            TestGraph::MemoryGraph m2;

            // clang-format off
            m2.m_nodes.push_back(
                TestGraph::Node{.m_id = 1,
                                .m_type = 1,
                                .m_neighbors{std::vector<TestGraph::NeighborRecord>{{8, 0, 1.5f},
                                                                                    {9, 0, 3.0f},
                                                                                    {10, 1, 1.0f},
                                                                                    {11, 1, 1.0f},
                                                                                    {12, 1, 1.0f},
                                                                                    {13, 1, 1.0f},
                                                                                    {14, 1, 1.0f}}}});
            // clang-format on

            m2.m_watermark = 3;
            for (uint32_t i = 0; i < 7; ++i)
            {
                m2.m_edge_timestamps.emplace_back(ts_pair{1, -1});
            }

            // Initialize Graph
            TestGraph::convert(path, "0_0", std::move(m1), 2);
            TestGraph::convert(path, "1_0", std::move(m2), 2);
            snark::Metadata metadata(path.string());
            m_live_memory_graph = std::make_unique<snark::Graph>(
                std::move(metadata), std::vector<std::string>{path.string(), path.string()},
                std::vector<uint32_t>{0, 1}, snark::PartitionStorageType::memory);
        }
        {
            // Reuse data from the graph above in memory graph.
            std::vector<std::shared_ptr<grpc::Channel>> channels;
            auto path = m_path / "live_edges";

            for (uint32_t server_index = 0; server_index < 2; ++server_index)
            {
                auto service = std::make_shared<snark::GraphEngineServiceImpl>(
                    snark::Metadata(path.string()), std::vector<std::string>{path.string()},
                    std::vector<uint32_t>{server_index}, snark::PartitionStorageType::memory);
                m_live_memory_servers.push_back(std::make_shared<snark::GRPCServer>(
                    std::move(service), std::shared_ptr<snark::GraphSamplerServiceImpl>{}, std::string{"localhost:0"},
                    std::string{}, std::string{}, std::string{}));
                channels.emplace_back(m_live_memory_servers.back()->InProcessChannel());
            }

            m_live_distributed_graph = std::make_shared<snark::GRPCClient>(std::move(channels), 1, 1);
        }
    }

    void TearDown() override
    {
        // Disconnect client before shutting down servers(will happen automatically in destructors).
        m_distributed_graph.reset();
        m_live_distributed_graph.reset();

        std::filesystem::remove_all(m_path);
    }
};

// Neighbor Count Tests
TEST_F(TemporalTest, GetNeigborCountSinglePartition)
{
    // Check for singe edge type filter
    std::vector<snark::NodeId> nodes = {0, 1};
    std::vector<snark::Type> types = {0};
    std::vector<uint64_t> output_neighbors_count(nodes.size());
    std::vector<snark::Timestamp> ts = {2, 2};

    m_single_partition_graph->NeighborCount(std::span(nodes), std::span(types), std::span(ts), output_neighbors_count);
    EXPECT_EQ(std::vector<uint64_t>({0, 0}), output_neighbors_count);

    ts = {0, 0};
    m_single_partition_graph->NeighborCount(std::span(nodes), std::span(types), std::span(ts), output_neighbors_count);
    EXPECT_EQ(std::vector<uint64_t>({2, 1}), output_neighbors_count);

    // Check for different singe edge type filter
    types = {1};
    ts = {2, 2};
    std::fill_n(output_neighbors_count.begin(), 2, -1);

    m_single_partition_graph->NeighborCount(std::span(nodes), std::span(types), std::span(ts), output_neighbors_count);
    EXPECT_EQ(std::vector<uint64_t>({0, 1}), output_neighbors_count);

    // Check for both edge types
    types = {0, 1};
    std::fill_n(output_neighbors_count.begin(), 2, -1);

    m_single_partition_graph->NeighborCount(std::span(nodes), std::span(types), std::span(ts), output_neighbors_count);
    EXPECT_EQ(std::vector<uint64_t>({0, 1}), output_neighbors_count);

    // Check returns 0 for unsatisfying edge types
    types = {-1, 100};
    std::fill_n(output_neighbors_count.begin(), 2, -1);

    m_single_partition_graph->NeighborCount(std::span(nodes), std::span(types), std::span(ts), output_neighbors_count);
    EXPECT_EQ(std::vector<uint64_t>({0, 0}), output_neighbors_count);

    // Invalid node ids
    nodes = {99, 100};
    types = {0, 1};
    std::fill_n(output_neighbors_count.begin(), 2, -1);

    m_single_partition_graph->NeighborCount(std::span(nodes), std::span(types), std::span(ts), output_neighbors_count);
    EXPECT_EQ(std::vector<uint64_t>({0, 0}), output_neighbors_count);
}

TEST_F(TemporalTest, GetFullNeighborSinglePartition)
{
    // Check for singe edge type filter
    std::vector<snark::NodeId> nodes = {0, 1};
    std::vector<snark::Type> types = {0};
    std::vector<uint64_t> output_neighbors_count(nodes.size());
    std::vector<snark::Timestamp> ts = {2, 2};

    std::vector<snark::NodeId> output_neighbor_ids;
    std::vector<snark::Type> output_neighbor_types;
    std::vector<float> output_neighbors_weights;
    std::vector<snark::Timestamp> output_edge_created_ts;
    m_single_partition_graph->FullNeighbor(true, std::span(nodes), std::span(types), std::span(ts), output_neighbor_ids,
                                           output_neighbor_types, output_neighbors_weights, output_edge_created_ts,
                                           std::span(output_neighbors_count));
    EXPECT_TRUE(output_neighbor_ids.empty());
    EXPECT_TRUE(output_neighbor_types.empty());
    EXPECT_TRUE(output_neighbors_weights.empty());
    EXPECT_TRUE(output_edge_created_ts.empty());
    EXPECT_EQ(std::vector<uint64_t>({0, 0}), output_neighbors_count);

    ts = {0, 0};
    m_single_partition_graph->FullNeighbor(true, std::span(nodes), std::span(types), std::span(ts), output_neighbor_ids,
                                           output_neighbor_types, output_neighbors_weights, output_edge_created_ts,
                                           std::span(output_neighbors_count));
    EXPECT_EQ(std::vector<snark::NodeId>({1, 2, 3}), output_neighbor_ids);
    output_neighbor_ids.clear();
    EXPECT_EQ(std::vector<snark::Type>({0, 0, 0}), output_neighbor_types);
    output_neighbor_types.clear();
    EXPECT_EQ(std::vector<float>({1.f, 2.f, 1.f}), output_neighbors_weights);
    output_neighbors_weights.clear();
    EXPECT_EQ(std::vector<snark::Timestamp>({0, 0, 0}), output_edge_created_ts);
    output_edge_created_ts.clear();
    EXPECT_EQ(std::vector<uint64_t>({2, 1}), output_neighbors_count);
    std::fill_n(output_neighbors_count.begin(), 2, -1);

    // Check for different singe edge type filter
    types = {1};
    ts = {2, 2};
    m_single_partition_graph->FullNeighbor(true, std::span(nodes), std::span(types), std::span(ts), output_neighbor_ids,
                                           output_neighbor_types, output_neighbors_weights, output_edge_created_ts,
                                           std::span(output_neighbors_count));
    EXPECT_EQ(std::vector<snark::NodeId>({5}), output_neighbor_ids);
    output_neighbor_ids.clear();
    EXPECT_EQ(std::vector<snark::Type>({1}), output_neighbor_types);
    output_neighbor_types.clear();
    EXPECT_EQ(std::vector<float>({7.f}), output_neighbors_weights);
    output_neighbors_weights.clear();
    EXPECT_EQ(std::vector<snark::Timestamp>({2}), output_edge_created_ts);
    output_edge_created_ts.clear();
    EXPECT_EQ(std::vector<uint64_t>({0, 1}), output_neighbors_count);
    std::fill_n(output_neighbors_count.begin(), 2, -1);

    // Check for both edge types
    types = {0, 1};
    m_single_partition_graph->FullNeighbor(true, std::span(nodes), std::span(types), std::span(ts), output_neighbor_ids,
                                           output_neighbor_types, output_neighbors_weights, output_edge_created_ts,
                                           std::span(output_neighbors_count));
    EXPECT_EQ(std::vector<snark::NodeId>({5}), output_neighbor_ids);
    output_neighbor_ids.clear();
    EXPECT_EQ(std::vector<snark::Type>({1}), output_neighbor_types);
    output_neighbor_types.clear();
    EXPECT_EQ(std::vector<float>({7.f}), output_neighbors_weights);
    output_neighbors_weights.clear();
    EXPECT_EQ(std::vector<snark::Timestamp>({2}), output_edge_created_ts);
    output_edge_created_ts.clear();
    EXPECT_EQ(std::vector<uint64_t>({0, 1}), output_neighbors_count);
    std::fill_n(output_neighbors_count.begin(), 2, -1);

    // Check returns 0 for unsatisfying edge types
    types = {-1, 100};
    std::fill_n(output_neighbors_count.begin(), 2, -1);

    m_single_partition_graph->FullNeighbor(true, std::span(nodes), std::span(types), std::span(ts), output_neighbor_ids,
                                           output_neighbor_types, output_neighbors_weights, output_edge_created_ts,
                                           std::span(output_neighbors_count));
    EXPECT_TRUE(output_neighbor_ids.empty());
    EXPECT_TRUE(output_neighbor_types.empty());
    EXPECT_TRUE(output_neighbors_weights.empty());
    EXPECT_TRUE(output_edge_created_ts.empty());
    EXPECT_EQ(std::vector<uint64_t>({0, 0}), output_neighbors_count);

    // Invalid node ids
    nodes = {99, 100};
    types = {0, 1};
    std::fill_n(output_neighbors_count.begin(), 2, -1);

    m_single_partition_graph->FullNeighbor(true, std::span(nodes), std::span(types), std::span(ts), output_neighbor_ids,
                                           output_neighbor_types, output_neighbors_weights, output_edge_created_ts,
                                           std::span(output_neighbors_count));
    EXPECT_TRUE(output_neighbor_ids.empty());
    EXPECT_TRUE(output_neighbor_types.empty());
    EXPECT_TRUE(output_neighbors_weights.empty());
    EXPECT_TRUE(output_edge_created_ts.empty());
    EXPECT_EQ(std::vector<uint64_t>({0, 0}), output_neighbors_count);
}

TEST_F(TemporalTest, GetNeigborCountMultiplePartitions)
{
    // Check for singe edge type filter
    std::vector<snark::NodeId> nodes = {0, 1};
    std::vector<snark::Type> types = {1};
    std::vector<uint64_t> output_neighbors_count(nodes.size());

    m_multi_partition_graph->NeighborCount(std::span(nodes), std::span(types), {}, output_neighbors_count);
    EXPECT_EQ(std::vector<uint64_t>({0, 3}), output_neighbors_count);

    // Check for multiple edge types
    types = {0, 1};
    std::fill_n(output_neighbors_count.begin(), 2, -1);

    std::vector<snark::Timestamp> ts = {2, 2};
    m_multi_partition_graph->NeighborCount(std::span(nodes), std::span(types), std::span(ts), output_neighbors_count);
    EXPECT_EQ(std::vector<uint64_t>({0, 3}), output_neighbors_count);

    // Check non-existent edge types functionality
    types = {-1, 100};
    std::fill_n(output_neighbors_count.begin(), 2, -1);

    m_multi_partition_graph->NeighborCount(std::span(nodes), std::span(types), std::span(ts), output_neighbors_count);
    EXPECT_EQ(std::vector<uint64_t>({0, 0}), output_neighbors_count);

    // Check invalid node ids handling
    nodes = {99, 100};
    types = {0, 1};
    std::fill_n(output_neighbors_count.begin(), 2, -1);

    m_multi_partition_graph->NeighborCount(std::span(nodes), std::span(types), std::span(ts), output_neighbors_count);
    EXPECT_EQ(std::vector<uint64_t>({0, 0}), output_neighbors_count);
}

TEST_F(TemporalTest, GetFullNeighborMultiplePartitions)
{
    // Check for singe edge type filter
    std::vector<snark::NodeId> nodes = {0, 1};
    std::vector<snark::Type> types = {1};
    std::vector<uint64_t> output_neighbors_count(nodes.size());
    std::vector<snark::Timestamp> ts = {2, 2};

    std::vector<snark::NodeId> output_neighbor_ids;
    std::vector<snark::Type> output_neighbor_types;
    std::vector<float> output_neighbors_weights;
    std::vector<snark::Timestamp> output_edge_created_ts;
    m_multi_partition_graph->FullNeighbor(true, std::span(nodes), std::span(types), std::span(ts), output_neighbor_ids,
                                          output_neighbor_types, output_neighbors_weights, output_edge_created_ts,
                                          std::span(output_neighbors_count));
    EXPECT_EQ(std::vector<snark::NodeId>({5, 7}), output_neighbor_ids);
    output_neighbor_ids.clear();
    EXPECT_EQ(std::vector<snark::Type>({1, 1}), output_neighbor_types);
    output_neighbor_types.clear();
    EXPECT_EQ(std::vector<float>({1.f, 3.0f}), output_neighbors_weights);
    output_neighbors_weights.clear();
    EXPECT_EQ(std::vector<snark::Timestamp>({1, 2}), output_edge_created_ts);
    output_edge_created_ts.clear();
    EXPECT_EQ(std::vector<uint64_t>({0, 2}), output_neighbors_count);
    std::fill_n(output_neighbors_count.begin(), 2, -1);

    // Check for multiple edge types
    types = {0, 1};
    m_multi_partition_graph->FullNeighbor(true, std::span(nodes), std::span(types), std::span(ts), output_neighbor_ids,
                                          output_neighbor_types, output_neighbors_weights, output_edge_created_ts,
                                          std::span(output_neighbors_count));
    EXPECT_EQ(std::vector<snark::NodeId>({4, 5, 7}), output_neighbor_ids);
    output_neighbor_ids.clear();
    EXPECT_EQ(std::vector<snark::Type>({0, 1, 1}), output_neighbor_types);
    output_neighbor_types.clear();
    EXPECT_EQ(std::vector<float>({1.f, 1.f, 3.f}), output_neighbors_weights);
    output_neighbors_weights.clear();
    EXPECT_EQ(std::vector<snark::Timestamp>({1, 1, 2}), output_edge_created_ts);
    output_edge_created_ts.clear();
    EXPECT_EQ(std::vector<uint64_t>({0, 3}), output_neighbors_count);
    std::fill_n(output_neighbors_count.begin(), 2, -1);

    // Check for different singe edge type filter
    types = {0};
    ts = {2, 2};
    m_multi_partition_graph->FullNeighbor(true, std::span(nodes), std::span(types), std::span(ts), output_neighbor_ids,
                                          output_neighbor_types, output_neighbors_weights, output_edge_created_ts,
                                          std::span(output_neighbors_count));
    EXPECT_EQ(std::vector<snark::NodeId>({4}), output_neighbor_ids);
    output_neighbor_ids.clear();
    EXPECT_EQ(std::vector<snark::Type>({0}), output_neighbor_types);
    output_neighbor_types.clear();
    EXPECT_EQ(std::vector<float>({1.f}), output_neighbors_weights);
    output_neighbors_weights.clear();
    EXPECT_EQ(std::vector<snark::Timestamp>({1}), output_edge_created_ts);
    output_edge_created_ts.clear();
    EXPECT_EQ(std::vector<uint64_t>({0, 1}), output_neighbors_count);
    std::fill_n(output_neighbors_count.begin(), 2, -1);

    // Check returns 0 for unsatisfying edge types
    types = {-1, 100};
    std::fill_n(output_neighbors_count.begin(), 2, -1);

    m_multi_partition_graph->FullNeighbor(true, std::span(nodes), std::span(types), std::span(ts), output_neighbor_ids,
                                          output_neighbor_types, output_neighbors_weights, output_edge_created_ts,
                                          std::span(output_neighbors_count));
    EXPECT_TRUE(output_neighbor_ids.empty());
    EXPECT_TRUE(output_neighbor_types.empty());
    EXPECT_TRUE(output_neighbors_weights.empty());
    EXPECT_TRUE(output_edge_created_ts.empty());
    EXPECT_EQ(std::vector<uint64_t>({0, 0}), output_neighbors_count);

    // Invalid node ids
    nodes = {99, 100};
    types = {0, 1};
    std::fill_n(output_neighbors_count.begin(), 2, -1);

    m_multi_partition_graph->FullNeighbor(true, std::span(nodes), std::span(types), std::span(ts), output_neighbor_ids,
                                          output_neighbor_types, output_neighbors_weights, output_edge_created_ts,
                                          std::span(output_neighbors_count));
    EXPECT_TRUE(output_neighbor_ids.empty());
    EXPECT_TRUE(output_neighbor_types.empty());
    EXPECT_TRUE(output_neighbors_weights.empty());
    EXPECT_TRUE(output_edge_created_ts.empty());
    EXPECT_EQ(std::vector<uint64_t>({0, 0}), output_neighbors_count);
}

TEST_F(TemporalTest, GetSampleNeighborsMultiplePartitions)
{
    // Check for singe edge type filter
    std::vector<snark::NodeId> nodes = {0, 1};
    std::vector<snark::Type> types = {1};
    std::vector<snark::Timestamp> ts = {2, 2};
    size_t sample_count = 2;

    std::vector<snark::NodeId> output_neighbor_ids(sample_count * nodes.size());
    std::vector<snark::Type> output_neighbor_types(sample_count * nodes.size());
    std::vector<float> output_neighbors_weights(sample_count * nodes.size());
    std::vector<float> output_neighbors_total_weights(nodes.size());
    std::vector<snark::Timestamp> output_edge_created_ts(sample_count * nodes.size(), -2);
    m_multi_partition_graph->SampleNeighbor(
        true, 33, std::span(nodes), std::span(types), std::span(ts), sample_count, std::span(output_neighbor_ids),
        std::span(output_neighbor_types), std::span(output_neighbors_weights),
        std::span(output_neighbors_total_weights), std::span(output_edge_created_ts), 42, 0.5f, 13);

    // Only available neighbor based on time/type is 5
    EXPECT_EQ(std::vector<snark::NodeId>({42, 42, 5, 5}), output_neighbor_ids);
    std::fill(std::begin(output_neighbor_ids), std::end(output_neighbor_ids), -1);
    EXPECT_EQ(std::vector<snark::Type>({13, 13, 1, 1}), output_neighbor_types);
    std::fill(std::begin(output_neighbor_types), std::end(output_neighbor_types), -1);
    EXPECT_EQ(std::vector<float>({0.5f, 0.5f, 1.f, 1.f}), output_neighbors_weights);
    std::fill(std::begin(output_neighbors_weights), std::end(output_neighbors_weights), -1);
    EXPECT_EQ(std::vector<float>({0.f, 4.0f}), output_neighbors_total_weights);
    std::fill(std::begin(output_neighbors_total_weights), std::end(output_neighbors_total_weights), 0);
    EXPECT_EQ(std::vector<snark::Timestamp>({snark::PLACEHOLDER_TIMESTAMP, snark::PLACEHOLDER_TIMESTAMP, 1, 1}),
              output_edge_created_ts);
    std::fill(std::begin(output_edge_created_ts), std::end(output_edge_created_ts), -2);

    // Check for multiple edge types
    types = {0, 1};
    m_multi_partition_graph->SampleNeighbor(
        true, 36, std::span(nodes), std::span(types), std::span(ts), sample_count, std::span(output_neighbor_ids),
        std::span(output_neighbor_types), std::span(output_neighbors_weights),
        std::span(output_neighbors_total_weights), std::span(output_edge_created_ts), 42, 0.5f, 13);
    EXPECT_EQ(std::vector<snark::NodeId>({42, 42, 7, 5}), output_neighbor_ids);
    std::fill(std::begin(output_neighbor_ids), std::end(output_neighbor_ids), -1);
    EXPECT_EQ(std::vector<snark::Type>({13, 13, 1, 1}), output_neighbor_types);
    std::fill(std::begin(output_neighbor_types), std::end(output_neighbor_types), -1);
    EXPECT_EQ(std::vector<float>({0.5f, 0.5f, 3.f, 1.f}), output_neighbors_weights);
    std::fill(std::begin(output_neighbors_weights), std::end(output_neighbors_weights), -1);
    EXPECT_EQ(std::vector<float>({0.f, 5.f}), output_neighbors_total_weights);
    std::fill(std::begin(output_neighbors_total_weights), std::end(output_neighbors_total_weights), 0);
    EXPECT_EQ(std::vector<snark::Timestamp>({snark::PLACEHOLDER_TIMESTAMP, snark::PLACEHOLDER_TIMESTAMP, 2, 1}),
              output_edge_created_ts);
    std::fill(std::begin(output_edge_created_ts), std::end(output_edge_created_ts), -2);

    // Check for different singe edge type filter
    types = {1};
    ts = {0, 0};
    m_multi_partition_graph->SampleNeighbor(
        true, 37, std::span(nodes), std::span(types), std::span(ts), sample_count, std::span(output_neighbor_ids),
        std::span(output_neighbor_types), std::span(output_neighbors_weights),
        std::span(output_neighbors_total_weights), std::span(output_edge_created_ts), 42, 0.5f, 13);
    EXPECT_EQ(std::vector<snark::NodeId>({42, 42, 6, 6}), output_neighbor_ids);
    std::fill(std::begin(output_neighbor_ids), std::end(output_neighbor_ids), -1);
    EXPECT_EQ(std::vector<snark::Type>({13, 13, 1, 1}), output_neighbor_types);
    std::fill(std::begin(output_neighbor_types), std::end(output_neighbor_types), -1);
    EXPECT_EQ(std::vector<float>({0.5f, 0.5f, 1.5f, 1.5f}), output_neighbors_weights);
    std::fill(std::begin(output_neighbors_weights), std::end(output_neighbors_weights), -1);
    EXPECT_EQ(std::vector<float>({0.f, 1.5f}), output_neighbors_total_weights);
    std::fill(std::begin(output_neighbors_total_weights), std::end(output_neighbors_total_weights), 0);
    EXPECT_EQ(std::vector<snark::Timestamp>({snark::PLACEHOLDER_TIMESTAMP, snark::PLACEHOLDER_TIMESTAMP, 0, 0}),
              output_edge_created_ts);
    std::fill(std::begin(output_edge_created_ts), std::end(output_edge_created_ts), -2);

    // Check returns 0 for unsatisfying edge types
    types = {-1, 100};

    m_multi_partition_graph->SampleNeighbor(
        true, 33, std::span(nodes), std::span(types), std::span(ts), sample_count, std::span(output_neighbor_ids),
        std::span(output_neighbor_types), std::span(output_neighbors_weights),
        std::span(output_neighbors_total_weights), std::span(output_edge_created_ts), 42, 0.5f, 13);
    EXPECT_EQ(std::vector<snark::NodeId>({42, 42, 42, 42}), output_neighbor_ids);
    std::fill(std::begin(output_neighbor_ids), std::end(output_neighbor_ids), -1);
    EXPECT_EQ(std::vector<snark::Type>({13, 13, 13, 13}), output_neighbor_types);
    std::fill(std::begin(output_neighbor_types), std::end(output_neighbor_types), -1);
    EXPECT_EQ(std::vector<float>({0.5f, 0.5f, 0.5f, 0.5f}), output_neighbors_weights);
    std::fill(std::begin(output_neighbors_weights), std::end(output_neighbors_weights), -1);
    EXPECT_EQ(std::vector<float>({0.f, 0.f}), output_neighbors_total_weights);
    std::fill(std::begin(output_neighbors_total_weights), std::end(output_neighbors_total_weights), 0);
    EXPECT_EQ(std::vector<snark::Timestamp>(4, snark::PLACEHOLDER_TIMESTAMP), output_edge_created_ts);
    std::fill(std::begin(output_edge_created_ts), std::end(output_edge_created_ts), -2);

    // Invalid node ids
    nodes = {99, 100};
    types = {0, 1};

    m_multi_partition_graph->SampleNeighbor(
        true, 33, std::span(nodes), std::span(types), std::span(ts), sample_count, std::span(output_neighbor_ids),
        std::span(output_neighbor_types), std::span(output_neighbors_weights),
        std::span(output_neighbors_total_weights), std::span(output_edge_created_ts), 42, 0.5f, 13);

    EXPECT_EQ(std::vector<snark::NodeId>({42, 42, 42, 42}), output_neighbor_ids);
    EXPECT_EQ(std::vector<snark::Type>({13, 13, 13, 13}), output_neighbor_types);
    EXPECT_EQ(std::vector<float>({0.5f, 0.5f, 0.5f, 0.5f}), output_neighbors_weights);
    EXPECT_EQ(std::vector<float>({0.f, 0.f}), output_neighbors_total_weights);
    EXPECT_EQ(std::vector<snark::Timestamp>(4, snark::PLACEHOLDER_TIMESTAMP), output_edge_created_ts);
}

TEST_F(TemporalTest, GetUniformSampleNeighborMultiplePartitions)
{
    // Check for singe edge type filter
    std::vector<snark::NodeId> nodes = {0, 1};
    std::vector<snark::Type> types = {1};
    std::vector<snark::Timestamp> ts = {2, 2};
    size_t sample_count = 2;

    std::vector<snark::NodeId> output_neighbor_ids(sample_count * nodes.size());
    std::vector<snark::Type> output_neighbor_types(sample_count * nodes.size());
    std::vector<uint64_t> output_neighbors_total_counts(nodes.size());
    std::vector<snark::Timestamp> output_edge_created_ts(sample_count * nodes.size(), -2);
    m_multi_partition_graph->UniformSampleNeighbor(
        false, true, 33, std::span(nodes), std::span(types), std::span(ts), sample_count,
        std::span(output_neighbor_ids), std::span(output_neighbor_types), std::span(output_neighbors_total_counts),
        std::span(output_edge_created_ts), 42, 13);

    // Only available neighbors based on time/type are 5 and 7
    EXPECT_EQ(std::vector<snark::NodeId>({42, 42, 7, 5}), output_neighbor_ids);
    std::fill(std::begin(output_neighbor_ids), std::end(output_neighbor_ids), -1);
    EXPECT_EQ(std::vector<snark::Type>({13, 13, 1, 1}), output_neighbor_types);
    std::fill(std::begin(output_neighbor_types), std::end(output_neighbor_types), -1);
    std::fill(std::begin(output_neighbors_total_counts), std::end(output_neighbors_total_counts), 0);
    EXPECT_EQ(std::vector<snark::Timestamp>({snark::PLACEHOLDER_TIMESTAMP, snark::PLACEHOLDER_TIMESTAMP, 2, 1}),
              output_edge_created_ts);
    std::fill(std::begin(output_edge_created_ts), std::end(output_edge_created_ts), -2);

    m_multi_partition_graph->UniformSampleNeighbor(
        true, true, 33, std::span(nodes), std::span(types), std::span(ts), sample_count, std::span(output_neighbor_ids),
        std::span(output_neighbor_types), std::span(output_neighbors_total_counts), std::span(output_edge_created_ts),
        42, 13);

    EXPECT_EQ(std::vector<snark::NodeId>({42, 42, 5, 7}), output_neighbor_ids);
    std::fill(std::begin(output_neighbor_ids), std::end(output_neighbor_ids), -1);
    EXPECT_EQ(std::vector<snark::Type>({13, 13, 1, 1}), output_neighbor_types);
    std::fill(std::begin(output_neighbor_types), std::end(output_neighbor_types), -1);
    EXPECT_EQ(std::vector<uint64_t>({0, 2}), output_neighbors_total_counts);
    std::fill(std::begin(output_neighbors_total_counts), std::end(output_neighbors_total_counts), 0);
    EXPECT_EQ(std::vector<snark::Timestamp>({snark::PLACEHOLDER_TIMESTAMP, snark::PLACEHOLDER_TIMESTAMP, 1, 2}),
              output_edge_created_ts);
    std::fill(std::begin(output_edge_created_ts), std::end(output_edge_created_ts), -2);

    ts = {1, 1};
    m_multi_partition_graph->UniformSampleNeighbor(
        false, true, 33, std::span(nodes), std::span(types), std::span(ts), sample_count,
        std::span(output_neighbor_ids), std::span(output_neighbor_types), std::span(output_neighbors_total_counts),
        std::span(output_edge_created_ts), 42, 13);
    // Only available neighbors based on time/type is 5
    EXPECT_EQ(std::vector<snark::NodeId>({42, 42, 5, 5}), output_neighbor_ids);
    std::fill(std::begin(output_neighbor_ids), std::end(output_neighbor_ids), -1);
    EXPECT_EQ(std::vector<snark::Type>({13, 13, 1, 1}), output_neighbor_types);
    std::fill(std::begin(output_neighbor_types), std::end(output_neighbor_types), -1);
    EXPECT_EQ(std::vector<uint64_t>({0, 1}), output_neighbors_total_counts);
    std::fill(std::begin(output_neighbors_total_counts), std::end(output_neighbors_total_counts), 0);
    EXPECT_EQ(std::vector<snark::Timestamp>({snark::PLACEHOLDER_TIMESTAMP, snark::PLACEHOLDER_TIMESTAMP, 1, 1}),
              output_edge_created_ts);
    std::fill(std::begin(output_edge_created_ts), std::end(output_edge_created_ts), -2);

    m_multi_partition_graph->UniformSampleNeighbor(
        true, true, 33, std::span(nodes), std::span(types), std::span(ts), sample_count, std::span(output_neighbor_ids),
        std::span(output_neighbor_types), std::span(output_neighbors_total_counts), std::span(output_edge_created_ts),
        42, 13);

    EXPECT_EQ(std::vector<snark::NodeId>({42, 42, 5, 42}), output_neighbor_ids);
    std::fill(std::begin(output_neighbor_ids), std::end(output_neighbor_ids), -1);
    EXPECT_EQ(std::vector<snark::Type>({13, 13, 1, 13}), output_neighbor_types);
    std::fill(std::begin(output_neighbor_types), std::end(output_neighbor_types), -1);
    EXPECT_EQ(std::vector<uint64_t>({0, 1}), output_neighbors_total_counts);
    std::fill(std::begin(output_neighbors_total_counts), std::end(output_neighbors_total_counts), 0);
    EXPECT_EQ(std::vector<snark::Timestamp>(
                  {snark::PLACEHOLDER_TIMESTAMP, snark::PLACEHOLDER_TIMESTAMP, 1, snark::PLACEHOLDER_TIMESTAMP}),
              output_edge_created_ts);
    std::fill(std::begin(output_edge_created_ts), std::end(output_edge_created_ts), -2);

    // Check for multiple edge types
    types = {0, 1};
    ts = {0, 0};
    m_multi_partition_graph->UniformSampleNeighbor(
        false, true, 36, std::span(nodes), std::span(types), std::span(ts), sample_count,
        std::span(output_neighbor_ids), std::span(output_neighbor_types), std::span(output_neighbors_total_counts),
        std::span(output_edge_created_ts), 42, 13);
    EXPECT_EQ(std::vector<snark::NodeId>({1, 2, 6, 3}), output_neighbor_ids);
    EXPECT_EQ(std::vector<snark::Type>({0, 0, 1, 0}), output_neighbor_types);
    EXPECT_EQ(std::vector<uint64_t>({2, 2}), output_neighbors_total_counts);
    EXPECT_EQ(std::vector<snark::Timestamp>({0, 0, 0, 0}), output_edge_created_ts);
}

TEST_F(TemporalTest, StatisticalNeighborSamplingWithoutDeletionsSinglePartition)
{
    std::vector<snark::NodeId> nodes = {0};
    std::vector<snark::Type> types = {0, 1};
    std::vector<uint64_t> output_neighbors_count(nodes.size());
    std::vector<snark::Timestamp> ts = {0};

    size_t sample_count = 3;
    std::vector<snark::NodeId> output_neighbor_ids(sample_count);
    std::vector<snark::Type> output_neighbor_types(sample_count);
    std::vector<float> output_neighbors_weights(sample_count);
    std::vector<float> output_neighbors_total_weights(sample_count);
    std::vector<snark::Timestamp> output_edge_created_ts(sample_count);
    std::vector<size_t> output_neighbor_ids_count(7, 0);
    for (size_t i = 0; i < 1000; i++)
    {
        m_live_memory_graph->SampleNeighbor(
            false, i * i, std::span(nodes), std::span(types), std::span(ts), sample_count,
            std::span(output_neighbor_ids), std::span(output_neighbor_types), std::span(output_neighbors_weights),
            std::span(output_neighbors_total_weights), std::span(output_edge_created_ts), 42, 13, 0);
        for (size_t j = 0; j < output_neighbor_ids.size(); j++)
        {
            output_neighbor_ids_count[output_neighbor_ids[j]]++;
        }

        std::fill(std::begin(output_neighbors_weights), std::end(output_neighbors_weights), 0.0);
        std::fill(std::begin(output_neighbors_total_weights), std::end(output_neighbors_total_weights), 0.0);
    }

    // Frequency of each node should be proportional to its edge weight in that partition
    EXPECT_EQ(std::vector<size_t>({0, 198, 435, 570, 794, 1003, 0}), output_neighbor_ids_count);
}

template <typename GraphType> std::vector<size_t> UniformSamplerHelper(GraphType *graph, bool with_replacement)
{
    std::vector<snark::NodeId> nodes = {0, 1};
    std::vector<snark::Type> types = {0, 1};
    std::vector<snark::Timestamp> ts = {1, 1};

    size_t sample_count = 3;
    std::vector<uint64_t> output_neighbors_count(nodes.size());
    std::vector<snark::NodeId> output_neighbor_ids(sample_count * nodes.size());
    std::vector<snark::Type> output_neighbor_types(sample_count * nodes.size());
    std::vector<uint64_t> output_neighbors_counts(sample_count * nodes.size());
    std::vector<snark::Timestamp> output_edge_created_ts(sample_count * nodes.size());
    std::vector<size_t> output_neighbor_ids_count(15, 0);
    for (size_t i = 0; i < 1000; i++)
    {
        if constexpr (std::is_same_v<GraphType, snark::Graph>)
        {
            graph->UniformSampleNeighbor(with_replacement, false, i * i, std::span(nodes), std::span(types),
                                         std::span(ts), sample_count, std::span(output_neighbor_ids),
                                         std::span(output_neighbor_types), std::span(output_neighbors_counts),
                                         std::span(output_edge_created_ts), 42, 13);
        }
        else if constexpr (std::is_same_v<GraphType, snark::GRPCClient>)
        {
            graph->UniformSampleNeighbor(with_replacement, false, i * i, std::span(nodes), std::span(types),
                                         std::span(ts), sample_count, std::span(output_neighbor_ids),
                                         std::span(output_neighbor_types), std::span(output_edge_created_ts), 42, 13);
        }

        for (size_t j = 0; j < output_neighbor_ids.size(); j++)
        {
            output_neighbor_ids_count[output_neighbor_ids[j]]++;
        }

        std::fill(std::begin(output_neighbors_counts), std::end(output_neighbors_counts), 0);
    }

    EXPECT_EQ(std::accumulate(std::begin(output_neighbor_ids_count), std::end(output_neighbor_ids_count), 0), 6000);
    return output_neighbor_ids_count;
}

TEST_F(TemporalTest, StatisticalUniformNeighborSamplingWithoutDeletions)
{
    std::vector<snark::NodeId> nodes = {1};
    std::vector<snark::Type> types = {0, 1};
    std::vector<uint64_t> output_neighbors_count(nodes.size());
    std::vector<snark::Timestamp> ts = {0};

    size_t sample_count = 3;
    std::vector<snark::NodeId> output_neighbor_ids(sample_count);
    std::vector<snark::Type> output_neighbor_types(sample_count);
    std::vector<size_t> output_neighbors_counts(sample_count);
    std::vector<snark::Timestamp> output_edge_created_ts(sample_count);
    std::vector<size_t> output_neighbor_ids_count(15, 0);
    for (size_t i = 0; i < 1000; i++)
    {
        m_live_memory_graph->UniformSampleNeighbor(false, false, i * i, std::span(nodes), std::span(types),
                                                   std::span(ts), sample_count, std::span(output_neighbor_ids),
                                                   std::span(output_neighbor_types), std::span(output_neighbors_counts),
                                                   std::span(output_edge_created_ts), 42, 13);
        for (size_t j = 0; j < output_neighbor_ids.size(); j++)
        {
            EXPECT_TRUE(output_neighbor_ids[j] >= 0 && output_neighbor_ids[j] < 15);
            if (output_neighbor_ids[j] >= 0 && output_neighbor_ids[j] < 15)
            {
                output_neighbor_ids_count[output_neighbor_ids[j]]++;
            }
        }

        std::fill(std::begin(output_neighbors_counts), std::end(output_neighbors_counts), 0);
    }

    // Only nodes from partition 1 are available. Expect a roughly equal split between 2 neighbors of 3000 total.
    EXPECT_EQ(std::vector<size_t>({0, 0, 0, 0, 0, 0, 1456, 1544, 0, 0, 0, 0, 0, 0, 0}), output_neighbor_ids_count);

    ts = {1, 1};
    nodes = {0, 1};
    output_neighbor_ids.resize(sample_count * nodes.size());
    output_neighbor_types.resize(sample_count * nodes.size());
    output_neighbors_counts.resize(sample_count * nodes.size());
    output_edge_created_ts.resize(sample_count * nodes.size());

    // First 5 neighbors belong to node 0 and should have frequency average of 1000*3/5 = 600.
    // Last 10 neighbors belong to node 1 and should have frequency average of 1000*3/9 = 330.
    EXPECT_EQ(UniformSamplerHelper(m_live_memory_graph.get(), false),
              std::vector<size_t>({0, 588, 621, 575, 593, 623, 352, 349, 310, 316, 324, 326, 377, 321, 325}));
    EXPECT_EQ(UniformSamplerHelper(m_live_memory_graph.get(), true),
              std::vector<size_t>({0, 633, 571, 602, 606, 588, 332, 323, 322, 349, 328, 340, 341, 350, 315}));

    // We can't easily return deterministic responses so for simplicity we'll check counts are within a reasonable
    // range. 1-5 should be within 10% of 600, 6-15 should be within 15% of 333.
    auto with_replacement_counts = UniformSamplerHelper(m_live_memory_graph.get(), true);
    auto without_replacement_counts = UniformSamplerHelper(m_live_memory_graph.get(), false);
    for (size_t i = 1; i < 15; i++)
    {
        if (i <= 5)
        {
            EXPECT_TRUE(without_replacement_counts[i] >= 540 && without_replacement_counts[i] <= 660);
            EXPECT_TRUE(with_replacement_counts[i] >= 540 && with_replacement_counts[i] <= 660);
        }
        else
        {
            EXPECT_TRUE(without_replacement_counts[i] >= 287 && without_replacement_counts[i] <= 380);
            EXPECT_TRUE(with_replacement_counts[i] >= 287 && with_replacement_counts[i] <= 380);
        }
    }
}

TEST_F(TemporalTest, GetFullNeighborDistributed)
{
    // Check for singe edge type filter
    std::vector<snark::NodeId> nodes = {0, 1};
    std::vector<snark::Type> types = {1};
    std::vector<uint64_t> output_neighbors_count(nodes.size());
    std::vector<snark::Timestamp> ts = {2, 2};

    std::vector<snark::NodeId> output_neighbor_ids;
    std::vector<snark::Type> output_neighbor_types;
    std::vector<float> output_neighbors_weights;
    std::vector<snark::Timestamp> output_edge_created_ts;
    m_distributed_graph->FullNeighbor(true, std::span(nodes), std::span(types), std::span(ts), output_neighbor_ids,
                                      output_neighbor_types, output_neighbors_weights, output_edge_created_ts,
                                      std::span(output_neighbors_count));
    EXPECT_EQ(std::vector<snark::NodeId>({5, 7}), output_neighbor_ids);
    EXPECT_EQ(std::vector<snark::Type>({1, 1}), output_neighbor_types);
    EXPECT_EQ(std::vector<float>({1.f, 3.0f}), output_neighbors_weights);
    EXPECT_EQ(std::vector<uint64_t>({0, 2}), output_neighbors_count);
    EXPECT_EQ(std::vector<snark::Timestamp>({1, 2}), output_edge_created_ts);
}

TEST_F(TemporalTest, GetNeighborCountDistributed)
{
    std::vector<snark::NodeId> nodes = {0, 1};
    std::vector<snark::Type> types = {1};
    std::vector<uint64_t> output_neighbors_count(nodes.size());
    std::vector<snark::Timestamp> ts = {2, 2};
    m_distributed_graph->NeighborCount(std::span(nodes), std::span(types), std::span(ts),
                                       std::span(output_neighbors_count));
    EXPECT_EQ(std::vector<uint64_t>({0, 2}), output_neighbors_count);

    types = {0, 1};
    m_distributed_graph->NeighborCount(std::span(nodes), std::span(types), std::span(ts),
                                       std::span(output_neighbors_count));
    EXPECT_EQ(std::vector<uint64_t>({0, 3}), output_neighbors_count);
}

TEST_F(TemporalTest, GetSampleNeighborsDistributed)
{
    std::vector<snark::NodeId> nodes = {0, 1};
    // Keep types = 0 to keep test stable and avoid merging neighbors from multiple shards.
    std::vector<snark::Type> types = {0};
    std::vector<snark::Timestamp> ts = {2, 2};
    size_t sample_count = 2;

    std::vector<snark::NodeId> output_neighbor_ids(sample_count * nodes.size());
    std::vector<snark::Type> output_neighbor_types(sample_count * nodes.size());
    std::vector<float> output_neighbors_weights(sample_count * nodes.size());
    std::vector<snark::Timestamp> output_edge_created_ts(sample_count * nodes.size(), -2);
    m_distributed_graph->WeightedSampleNeighbor(true, 37, std::span(nodes), std::span(types), std::span(ts),
                                                sample_count, std::span(output_neighbor_ids),
                                                std::span(output_neighbor_types), std::span(output_neighbors_weights),
                                                std::span(output_edge_created_ts), 42, 0.5f, 13);
    EXPECT_EQ(std::vector<snark::NodeId>({42, 42, 4, 4}), output_neighbor_ids);
    EXPECT_EQ(std::vector<snark::Type>({13, 13, 0, 0}), output_neighbor_types);
    EXPECT_EQ(std::vector<float>({0.5f, 0.5f, 1.f, 1.f}), output_neighbors_weights);
    EXPECT_EQ(std::vector<snark::Timestamp>({snark::PLACEHOLDER_TIMESTAMP, snark::PLACEHOLDER_TIMESTAMP, 1, 1}),
              output_edge_created_ts);
}

TEST_F(TemporalTest, GetUniformSampleNeighborsDistributed)
{
    std::vector<snark::NodeId> nodes = {0, 1};
    // Keep types = 0 to keep test stable and avoid merging neighbors from multiple shards.
    std::vector<snark::Type> types = {0};
    std::vector<snark::Timestamp> ts = {2, 2};
    size_t sample_count = 2;

    std::vector<snark::NodeId> output_neighbor_ids(sample_count * nodes.size());
    std::vector<snark::Type> output_neighbor_types(sample_count * nodes.size());
    std::vector<snark::Timestamp> output_edge_created_ts(sample_count * nodes.size(), -2);
    m_distributed_graph->UniformSampleNeighbor(
        true, true, 37, std::span(nodes), std::span(types), std::span(ts), sample_count, std::span(output_neighbor_ids),
        std::span(output_neighbor_types), std::span(output_edge_created_ts), 42, 13);
    EXPECT_EQ(std::vector<snark::NodeId>({42, 42, 4, 42}), output_neighbor_ids);
    EXPECT_EQ(std::vector<snark::Type>({13, 13, 0, 13}), output_neighbor_types);
    EXPECT_EQ(std::vector<snark::Timestamp>(
                  {snark::PLACEHOLDER_TIMESTAMP, snark::PLACEHOLDER_TIMESTAMP, 1, snark::PLACEHOLDER_TIMESTAMP}),
              output_edge_created_ts);

    std::fill(std::begin(output_neighbor_ids), std::end(output_neighbor_ids), -2);
    std::fill(std::begin(output_neighbor_types), std::end(output_neighbor_types), -2);
    std::fill(std::begin(output_edge_created_ts), std::end(output_edge_created_ts), -2);
    m_distributed_graph->UniformSampleNeighbor(
        false, true, 37, std::span(nodes), std::span(types), std::span(ts), sample_count,
        std::span(output_neighbor_ids), std::span(output_neighbor_types), std::span(output_edge_created_ts), 42, 13);
    EXPECT_EQ(std::vector<snark::NodeId>({42, 42, 4, 4}), output_neighbor_ids);
    EXPECT_EQ(std::vector<snark::Type>({13, 13, 0, 0}), output_neighbor_types);
    EXPECT_EQ(std::vector<snark::Timestamp>({snark::PLACEHOLDER_TIMESTAMP, snark::PLACEHOLDER_TIMESTAMP, 1, 1}),
              output_edge_created_ts);
}

namespace
{
template <typename Graph> void GetLastNCreatedNeighbors(Graph &g)
{
    std::vector<snark::NodeId> nodes = {0, 1};
    std::vector<snark::Type> types = {0};
    std::vector<snark::Timestamp> ts = {2, 2};
    size_t sample_count = 2;

    std::vector<snark::NodeId> output_neighbor_ids(sample_count * nodes.size());
    std::vector<snark::Type> output_neighbor_types(sample_count * nodes.size());
    std::vector<float> output_neighbors_weights(sample_count * nodes.size());
    std::vector<snark::Timestamp> output_timestamps(sample_count * nodes.size());
    g->LastNCreated(true, std::span(nodes), std::span(types), std::span(ts), sample_count,
                    std::span(output_neighbor_ids), std::span(output_neighbor_types),
                    std::span(output_neighbors_weights), std::span(output_timestamps), 42, 0.5f, 13, 99);
    EXPECT_EQ(std::vector<snark::NodeId>({42, 42, 4, 42}), output_neighbor_ids);
    EXPECT_EQ(std::vector<snark::Type>({13, 13, 0, 13}), output_neighbor_types);
    EXPECT_EQ(std::vector<float>({0.5f, 0.5f, 1.f, 0.5f}), output_neighbors_weights);
    EXPECT_EQ(std::vector<snark::Timestamp>({99, 99, 1, 99}), output_timestamps);
}
} // namespace

TEST_F(TemporalTest, GetLastNCreatedNeighborsDistributed)
{
    GetLastNCreatedNeighbors(m_distributed_graph);
}

TEST_F(TemporalTest, GetLastNCreatedNeighborsInMemory)
{
    GetLastNCreatedNeighbors(m_multi_partition_graph);
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
