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
    }

    void TearDown() override
    {
        // Disconnect client before shutting down servers(will happen automatically in destructors).
        m_distributed_graph.reset();

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
    m_single_partition_graph->FullNeighbor(std::span(nodes), std::span(types), std::span(ts), output_neighbor_ids,
                                           output_neighbor_types, output_neighbors_weights, output_edge_created_ts,
                                           std::span(output_neighbors_count));
    EXPECT_TRUE(output_neighbor_ids.empty());
    EXPECT_TRUE(output_neighbor_types.empty());
    EXPECT_TRUE(output_neighbors_weights.empty());
    EXPECT_TRUE(output_edge_created_ts.empty());
    EXPECT_EQ(std::vector<uint64_t>({0, 0}), output_neighbors_count);

    ts = {0, 0};
    m_single_partition_graph->FullNeighbor(std::span(nodes), std::span(types), std::span(ts), output_neighbor_ids,
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
    m_single_partition_graph->FullNeighbor(std::span(nodes), std::span(types), std::span(ts), output_neighbor_ids,
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
    m_single_partition_graph->FullNeighbor(std::span(nodes), std::span(types), std::span(ts), output_neighbor_ids,
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

    m_single_partition_graph->FullNeighbor(std::span(nodes), std::span(types), std::span(ts), output_neighbor_ids,
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

    m_single_partition_graph->FullNeighbor(std::span(nodes), std::span(types), std::span(ts), output_neighbor_ids,
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
    m_multi_partition_graph->FullNeighbor(std::span(nodes), std::span(types), std::span(ts), output_neighbor_ids,
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
    m_multi_partition_graph->FullNeighbor(std::span(nodes), std::span(types), std::span(ts), output_neighbor_ids,
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
    m_multi_partition_graph->FullNeighbor(std::span(nodes), std::span(types), std::span(ts), output_neighbor_ids,
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

    m_multi_partition_graph->FullNeighbor(std::span(nodes), std::span(types), std::span(ts), output_neighbor_ids,
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

    m_multi_partition_graph->FullNeighbor(std::span(nodes), std::span(types), std::span(ts), output_neighbor_ids,
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
        33, std::span(nodes), std::span(types), std::span(ts), sample_count, std::span(output_neighbor_ids),
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
        36, std::span(nodes), std::span(types), std::span(ts), sample_count, std::span(output_neighbor_ids),
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
        37, std::span(nodes), std::span(types), std::span(ts), sample_count, std::span(output_neighbor_ids),
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
        33, std::span(nodes), std::span(types), std::span(ts), sample_count, std::span(output_neighbor_ids),
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
        33, std::span(nodes), std::span(types), std::span(ts), sample_count, std::span(output_neighbor_ids),
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
        false, 33, std::span(nodes), std::span(types), std::span(ts), sample_count, std::span(output_neighbor_ids),
        std::span(output_neighbor_types), std::span(output_neighbors_total_counts), std::span(output_edge_created_ts),
        42, 13);

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
        true, 33, std::span(nodes), std::span(types), std::span(ts), sample_count, std::span(output_neighbor_ids),
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
        false, 33, std::span(nodes), std::span(types), std::span(ts), sample_count, std::span(output_neighbor_ids),
        std::span(output_neighbor_types), std::span(output_neighbors_total_counts), std::span(output_edge_created_ts),
        42, 13);
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
        true, 33, std::span(nodes), std::span(types), std::span(ts), sample_count, std::span(output_neighbor_ids),
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
        false, 36, std::span(nodes), std::span(types), std::span(ts), sample_count, std::span(output_neighbor_ids),
        std::span(output_neighbor_types), std::span(output_neighbors_total_counts), std::span(output_edge_created_ts),
        42, 13);
    EXPECT_EQ(std::vector<snark::NodeId>({1, 2, 6, 3}), output_neighbor_ids);
    EXPECT_EQ(std::vector<snark::Type>({0, 0, 1, 0}), output_neighbor_types);
    EXPECT_EQ(std::vector<uint64_t>({2, 2}), output_neighbors_total_counts);
    EXPECT_EQ(std::vector<snark::Timestamp>({0, 0, 0, 0}), output_edge_created_ts);
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
    m_distributed_graph->FullNeighbor(std::span(nodes), std::span(types), std::span(ts), output_neighbor_ids,
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
    m_distributed_graph->WeightedSampleNeighbor(37, std::span(nodes), std::span(types), std::span(ts), sample_count,
                                                std::span(output_neighbor_ids), std::span(output_neighbor_types),
                                                std::span(output_neighbors_weights), std::span(output_edge_created_ts),
                                                42, 0.5f, 13);
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
        true, 37, std::span(nodes), std::span(types), std::span(ts), sample_count, std::span(output_neighbor_ids),
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
        false, 37, std::span(nodes), std::span(types), std::span(ts), sample_count, std::span(output_neighbor_ids),
        std::span(output_neighbor_types), std::span(output_edge_created_ts), 42, 13);
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
    g->LastNCreated(std::span(nodes), std::span(types), std::span(ts), sample_count, std::span(output_neighbor_ids),
                    std::span(output_neighbor_types), std::span(output_neighbors_weights), std::span(output_timestamps),
                    42, 0.5f, 13, 99);
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
