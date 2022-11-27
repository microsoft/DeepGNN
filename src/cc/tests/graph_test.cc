// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

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

using WeightedNodePartitionList = std::vector<snark::WeightedNodeSamplerPartition>;
using WeightedEdgePartitionList = std::vector<snark::WeightedEdgeSamplerPartition>;
template <bool withRep> using UniformNodePartitionList = std::vector<snark::UniformNodeSamplerPartition<withRep>>;
template <bool withRep> using UniformEdgePartitionList = std::vector<snark::UniformEdgeSamplerPartition<withRep>>;

class StorageTypeGraphTest : public testing::TestWithParam<snark::PartitionStorageType>
{
};

TEST(GraphTest, NodeSamplingSingleType)
{
    snark::WeightedNodeSamplerPartition p1(
        std::vector<snark::WeightedNodeSamplerRecord>{snark::WeightedNodeSamplerRecord{0, 1, 0.1},
                                                      snark::WeightedNodeSamplerRecord{2, 3, 0.9}},
        0.2);
    snark::WeightedNodeSamplerPartition p2(
        std::vector<snark::WeightedNodeSamplerRecord>{snark::WeightedNodeSamplerRecord{4, 5, 0.1},
                                                      snark::WeightedNodeSamplerRecord{6, 7, 0.9}},
        1.0);
    auto partitions =
        std::make_shared<WeightedNodePartitionList>(WeightedNodePartitionList{std::move(p1), std::move(p2)});
    snark::WeightedNodeSampler s(std::vector<snark::Type>{0, 1},
                                 std::vector<std::shared_ptr<WeightedNodePartitionList>>{partitions});
    std::vector<snark::Type> inputs = {0};
    size_t count = 5;
    std::vector<snark::NodeId> nodes(count, -1);
    std::vector<snark::Type> types(count, -1);
    std::span input_nodes(nodes);
    std::span input_types(types);
    s.Sample(1, input_types, input_nodes);
    EXPECT_EQ(nodes, std::vector<snark::NodeId>({0, 2, 6, 6, 5}));
    EXPECT_EQ(types, std::vector<snark::Type>(count, 0));
}

TEST(GraphTest, NodeSamplingMultipleTypes)
{
    auto t0 = std::make_shared<WeightedNodePartitionList>(WeightedNodePartitionList{
        {std::vector<snark::WeightedNodeSamplerRecord>{snark::WeightedNodeSamplerRecord{0, 1, 0.1},
                                                       snark::WeightedNodeSamplerRecord{2, 3, 0.9}},
         1.0}});
    auto t1 = std::make_shared<WeightedNodePartitionList>(WeightedNodePartitionList{
        {std::vector<snark::WeightedNodeSamplerRecord>{snark::WeightedNodeSamplerRecord{4, 5, 0.1},
                                                       snark::WeightedNodeSamplerRecord{6, 7, 0.5},
                                                       snark::WeightedNodeSamplerRecord{8, 9, 0.5}},
         3.0}});

    snark::WeightedNodeSampler s(std::vector<snark::Type>{0, 1},
                                 std::vector<std::shared_ptr<WeightedNodePartitionList>>{t0, t1});
    size_t count = 4;
    std::vector<snark::NodeId> nodes(count, -1);
    std::vector<snark::Type> types(count, -1);
    std::span input_nodes(nodes);
    std::span input_types(types);
    s.Sample(42, input_types, input_nodes);
    EXPECT_EQ(nodes, std::vector<snark::NodeId>({2, 1, 8, 5}));
    EXPECT_EQ(types, std::vector<snark::Type>({0, 0, 1, 1}));
    EXPECT_EQ(s.Weight(), 4.f);
}

TEST(GraphTest, UniformNodeSamplingSingleType)
{
    snark::UniformNodeSamplerPartition<false> p1(std::vector<snark::NodeId>{0, 1, 2});
    snark::UniformNodeSamplerPartition<false> p2(std::vector<snark::NodeId>{3, 4, 5, 6, 7, 8, 9, 10});
    auto partitions = std::make_shared<UniformNodePartitionList<false>>(
        UniformNodePartitionList<false>{std::move(p1), std::move(p2)});
    snark::UniformNodeSamplerWithoutReplacement s(
        std::vector<snark::Type>{0, 1}, std::vector<std::shared_ptr<UniformNodePartitionList<false>>>{partitions});
    std::vector<snark::Type> inputs = {0};
    size_t count = 5;
    std::vector<snark::NodeId> nodes(count, -1);
    std::vector<snark::Type> types(count, -1);
    std::span input_nodes(nodes);
    std::span input_types(types);
    s.Sample(13, input_types, input_nodes);
    EXPECT_EQ(nodes, std::vector<snark::NodeId>({0, 1, 2, 3, 10}));
    EXPECT_EQ(types, std::vector<snark::Type>(count, 0));
}

TEST(GraphTest, UniformEdgeSamplingSingleType)
{
    snark::UniformEdgeSamplerPartition<false> p1(std::vector<std::pair<snark::NodeId, snark::NodeId>>{{0, 1}, {2, 11}});
    snark::UniformEdgeSamplerPartition<false> p2(
        std::vector<std::pair<snark::NodeId, snark::NodeId>>{{3, 4}, {5, 6}, {7, 8}, {9, 10}});
    auto partitions = std::make_shared<UniformEdgePartitionList<false>>(
        UniformEdgePartitionList<false>{std::move(p1), std::move(p2)});
    snark::UniformEdgeSamplerWithoutReplacement s(
        std::vector<snark::Type>{0, 1}, std::vector<std::shared_ptr<UniformEdgePartitionList<false>>>{partitions});
    std::vector<snark::Type> inputs = {0};
    size_t count = 5;
    std::vector<snark::NodeId> src(count, -1);
    std::vector<snark::NodeId> dst(count, -1);
    std::vector<snark::Type> types(count, -1);
    std::span input_src(src);
    std::span input_dst(dst);
    std::span input_types(types);
    s.Sample(13, input_types, input_src, input_dst);
    EXPECT_EQ(src, std::vector<snark::NodeId>({0, 2, 3, 5, 7}));
    EXPECT_EQ(dst, std::vector<snark::NodeId>({1, 11, 4, 6, 8}));
    EXPECT_EQ(types, std::vector<snark::Type>(count, 0));
}

TEST(GraphTest, UniformNodeSamplingRequestMoreThanNumberOfElements)
{
    snark::UniformNodeSamplerPartition<false> p1(std::vector<snark::NodeId>{0, 1, 2});
    snark::UniformNodeSamplerPartition<false> p2(std::vector<snark::NodeId>{3, 4, 5, 6, 7});
    auto partitions = std::make_shared<UniformNodePartitionList<false>>(
        UniformNodePartitionList<false>{std::move(p1), std::move(p2)});
    snark::UniformNodeSamplerWithoutReplacement s(
        std::vector<snark::Type>{0}, std::vector<std::shared_ptr<UniformNodePartitionList<false>>>{partitions});
    std::vector<snark::Type> inputs = {0};
    size_t count = 10;
    std::vector<snark::NodeId> nodes(count, -1);
    std::vector<snark::Type> types(count, -1);
    std::span input_nodes(nodes);
    std::span input_types(types);
    s.Sample(13, input_types, input_nodes);
    EXPECT_EQ(nodes, std::vector<snark::NodeId>({0, 1, 2, 3, 4, 5, 6, 7, -1, -1}));
    EXPECT_EQ(types, std::vector<snark::Type>({0, 0, 0, 0, 0, 0, 0, 0, -1, -1}));
}

TEST(GraphTest, UniformNodeSamplingMultipleTypes)
{
    auto t0 = std::make_shared<UniformNodePartitionList<false>>(
        UniformNodePartitionList<false>{snark::UniformNodeSamplerPartition<false>{std::vector<snark::NodeId>{0, 1}}});
    auto t1 = std::make_shared<UniformNodePartitionList<false>>(UniformNodePartitionList<false>{
        snark::UniformNodeSamplerPartition<false>{std::vector<snark::NodeId>{2, 3, 4, 5, 7}}});
    snark::UniformNodeSamplerWithoutReplacement s(
        std::vector<snark::Type>{0, 1}, std::vector<std::shared_ptr<UniformNodePartitionList<false>>>({t0, t1}));
    size_t count = 4;
    std::vector<snark::NodeId> nodes(count, -1);
    std::vector<snark::Type> types(count, -1);
    std::span input_nodes(nodes);
    std::span input_types(types);
    s.Sample(13, input_types, input_nodes);
    EXPECT_EQ(nodes, std::vector<snark::NodeId>({2, 7, 4, 5}));
    EXPECT_EQ(types, std::vector<snark::Type>({1, 1, 1, 1}));
}

template <bool with_replacement>
void RandomNodeSamplingStatisticalProperties(std::pair<size_t, size_t> minmax_0, std::pair<size_t, size_t> minmax_1)
{
    snark::NodeId curr_node = 0;
    const size_t num_types = 2;
    std::vector<std::vector<snark::NodeId>> records(num_types);
    const size_t nodes_per_type = 100;
    for (auto &t : records)
    {
        t.reserve(nodes_per_type);
        for (size_t i = 0; i < nodes_per_type; ++i)
        {
            t.emplace_back(curr_node++);
        }
    }
    std::vector<snark::Type> types;
    std::generate_n(std::back_inserter(types), num_types, [&types]() { return types.size(); });
    std::vector<std::shared_ptr<UniformNodePartitionList<with_replacement>>> partitions;
    for (auto t : records)
    {
        partitions.emplace_back();
        partitions.back() = std::make_shared<UniformNodePartitionList<with_replacement>>();
        partitions.back()->emplace_back(std::move(t));
    }

    snark::SamplerImpl<snark::UniformNodeSamplerPartition<with_replacement>> s(types, partitions);
    size_t batch_size = 16;
    std::vector<snark::NodeId> node_holder(batch_size, -1);
    std::vector<snark::Type> type_holder(batch_size, -1);
    std::vector<size_t> type_counts(num_types);
    std::vector<std::vector<size_t>> node_counts(num_types, std::vector<size_t>(nodes_per_type));
    for (int64_t seed = 1; seed < 10000; ++seed)
    {
        s.Sample(seed, std::span(type_holder), std::span(node_holder));
        for (size_t j = 0; j < batch_size; ++j)
        {
            ++type_counts[type_holder[j]];
            ++node_counts[type_holder[j]][node_holder[j] % nodes_per_type];
        }
    }

    // 79999 / 79985 ~ 50%
    EXPECT_EQ(type_counts, std::vector<size_t>({80004, 79980}));
    const auto [min_0, max_0] = std::minmax_element(std::begin(node_counts[0]), std::end(node_counts[0]));
    EXPECT_EQ(*min_0, minmax_0.first);
    EXPECT_EQ(*max_0, minmax_0.second);
    const auto [min_1, max_1] = std::minmax_element(std::begin(node_counts[1]), std::end(node_counts[1]));
    EXPECT_EQ(*min_1, minmax_1.first);
    EXPECT_EQ(*max_1, minmax_1.second);
}

TEST(GraphTest, RandomNodeSamplingWithReplacementStatisticalProperties)
{
    RandomNodeSamplingStatisticalProperties<true>({752, 847}, {734, 861});
}

TEST(GraphTest, RandomNodeSamplingWithoutReplacementStatisticalProperties)
{
    RandomNodeSamplingStatisticalProperties<false>({735, 863}, {740, 875});
}

TEST(GraphTest, EdgeSamplingSingleType)
{
    auto p1 = snark::WeightedEdgeSamplerPartition{
        std::vector<snark::WeightedEdgeSamplerRecord>{snark::WeightedEdgeSamplerRecord{0, 1, 2, 3, 0.1},
                                                      snark::WeightedEdgeSamplerRecord{4, 5, 6, 7, 0.9}},
        0.5};
    auto p2 = snark::WeightedEdgeSamplerPartition{
        std::vector<snark::WeightedEdgeSamplerRecord>{snark::WeightedEdgeSamplerRecord{8, 9, 10, 11, 0.1},
                                                      snark::WeightedEdgeSamplerRecord{12, 13, 14, 15, 0.5}},
        1.0};
    auto partitions =
        std::make_shared<WeightedEdgePartitionList>(WeightedEdgePartitionList{std::move(p1), std::move(p2)});
    snark::WeightedEdgeSampler s(std::vector<snark::Type>{0, 1},
                                 std::vector<std::shared_ptr<WeightedEdgePartitionList>>{partitions});
    std::vector<snark::Type> inputs = {0};
    size_t count = 5;
    std::vector<snark::NodeId> nodes_src(count, -1), nodes_dst(count, -1);
    std::vector<snark::Type> types(count, -1);
    std::span out_src(nodes_src), out_dst(nodes_dst);
    std::span out_types(types);
    s.Sample(42, out_types, out_src, out_dst);
    EXPECT_EQ(nodes_src, std::vector<snark::NodeId>({10, 14, 14, 14, 12}));
    EXPECT_EQ(nodes_dst, std::vector<snark::NodeId>({11, 15, 15, 15, 13}));
    EXPECT_EQ(types, std::vector<snark::Type>(count, 0));
}

TEST(GraphTest, EdgeSamplingMultipleTypes)
{
    auto t0 = std::make_shared<WeightedEdgePartitionList>(WeightedEdgePartitionList{
        {std::vector<snark::WeightedEdgeSamplerRecord>{snark::WeightedEdgeSamplerRecord{0, 1, 2, 3, 0.1},
                                                       snark::WeightedEdgeSamplerRecord{4, 5, 6, 7, 0.9}},
         1.0}});

    auto t1 = std::make_shared<WeightedEdgePartitionList>(WeightedEdgePartitionList{
        {std::vector<snark::WeightedEdgeSamplerRecord>{snark::WeightedEdgeSamplerRecord{8, 9, 10, 11, 0.1},
                                                       snark::WeightedEdgeSamplerRecord{12, 13, 14, 15, 0.5},
                                                       snark::WeightedEdgeSamplerRecord{16, 17, 18, 19, 0.5}},
         1.0}});
    snark::WeightedEdgeSampler s(std::vector<snark::Type>{0, 1},
                                 std::vector<std::shared_ptr<WeightedEdgePartitionList>>{t0, t1});

    std::vector<snark::Type> inputs = {0, 1};
    size_t count = 5;
    std::vector<snark::NodeId> nodes_src(count, -1), nodes_dst(count, -1);
    std::vector<snark::Type> types(count, -1);
    std::span out_src(nodes_src), out_dst(nodes_dst);
    std::span out_types(types);
    s.Sample(14, out_types, out_src, out_dst);
    EXPECT_EQ(nodes_src, std::vector<snark::NodeId>({4, 4, 10, 12, 10}));
    EXPECT_EQ(nodes_dst, std::vector<snark::NodeId>({5, 5, 11, 13, 11}));
    EXPECT_EQ(types, std::vector<snark::Type>({0, 0, 1, 1, 1}));
}

TEST(GraphTest, WeightSampleStatisticalProperties)
{
    snark::Type curr_type = 0;
    snark::Xoroshiro128PlusGenerator gen(42);
    const size_t num_types = 2;
    std::vector<std::vector<snark::WeightedNodeSamplerRecord>> records(num_types);
    const size_t nodes_per_type = 100;
    for (auto &t : records)
    {
        t.reserve(nodes_per_type);
        for (size_t i = 0; i < nodes_per_type; ++i)
        {
            // Keep alias sampling uniform for simplicity.
            t.emplace_back(snark::WeightedNodeSamplerRecord{
                snark::NodeId(i + curr_type * nodes_per_type),
                snark::NodeId((i + 1) % nodes_per_type + curr_type * nodes_per_type), 0.5f});
        }

        curr_type++;
    }
    std::vector<snark::Type> types;
    std::generate_n(std::back_inserter(types), num_types, [&types]() { return snark::Type(types.size()); });
    std::vector<std::shared_ptr<WeightedNodePartitionList>> partitions;
    std::array<float, num_types> type_weights = {0.3f, 0.7f};
    auto type_weight = std::begin(type_weights);
    for (auto t : records)
    {
        partitions.emplace_back();
        partitions.back() = std::make_shared<WeightedNodePartitionList>();
        partitions.back()->emplace_back(std::move(t), *type_weight);
        ++type_weight;
    }

    snark::SamplerImpl s(types, partitions);
    size_t batch_size = 9;
    std::vector<snark::NodeId> node_holder(batch_size, -1);
    std::vector<snark::Type> type_holder(batch_size, -1);
    int64_t seed = 42;
    std::vector<size_t> type_counts(num_types);
    std::vector<std::vector<size_t>> node_counts(num_types, std::vector<size_t>(nodes_per_type));
    for (size_t i = 0; i < 10000; ++i)
    {
        s.Sample(++seed, std::span(type_holder), std::span(node_holder));
        for (size_t j = 0; j < batch_size; ++j)
        {
            ++type_counts[type_holder[j]];
            ++node_counts[type_holder[j]][node_holder[j] % nodes_per_type];
        }
    }

    // 27008 / 62992 ~ 0.4288
    // 27000 / 63000 ~ 0.4285
    // 0.3 / 0.7 ~ 0.4285
    EXPECT_EQ(type_counts, std::vector<size_t>({27025, 62975}));

    const auto [min_0, max_0] = std::minmax_element(std::begin(node_counts[0]), std::end(node_counts[0]));
    const auto [min_1, max_1] = std::minmax_element(std::begin(node_counts[1]), std::end(node_counts[1]));
    // Counts should be fairly close to average 27003/nodes_per_type
    EXPECT_EQ(*min_0, 234);
    EXPECT_EQ(*max_0, 301);
    EXPECT_EQ(*min_1, 577);
    EXPECT_EQ(*max_1, 714);
}

TEST_P(StorageTypeGraphTest, NodeTypesMultipleNodes)
{
    TestGraph::MemoryGraph m;
    m.m_nodes.push_back(TestGraph::Node{.m_id = 0, .m_type = 0, .m_weight = 1.0f});
    m.m_nodes.push_back(TestGraph::Node{.m_id = 1, .m_type = 2, .m_weight = 1.0f});
    auto path = std::filesystem::temp_directory_path();
    TestGraph::convert(path, "0_0", std::move(m), 3);
    snark::Graph g(path.string(), std::vector<uint32_t>{0}, GetParam(), "");
    std::vector<snark::NodeId> nodes = {0, 1, 2};
    std::vector<snark::Type> output(3, -2);

    g.GetNodeType(std::span(nodes), std::span(output), -1);
    EXPECT_EQ(output, std::vector<snark::Type>({0, 2, -1}));
}

TEST_P(StorageTypeGraphTest, NodeFeaturesMultipleNodesSingleFeature)
{
    TestGraph::MemoryGraph m;
    std::vector<std::vector<float>> f1 = {std::vector<float>{1.0f, 2.0f, 3.0f}};
    std::vector<std::vector<float>> f2 = {std::vector<float>{5.0f, 6.0f, 7.0f}};
    m.m_nodes.push_back(TestGraph::Node{.m_id = 0, .m_type = 0, .m_weight = 1.0f, .m_float_features = f1});
    m.m_nodes.push_back(TestGraph::Node{.m_id = 1, .m_type = 1, .m_weight = 1.0f, .m_float_features = f2});
    auto path = std::filesystem::temp_directory_path();
    TestGraph::convert(path, "0_0", std::move(m), 2);
    snark::Graph g(path.string(), std::vector<uint32_t>{0}, GetParam(), "");
    std::vector<snark::NodeId> nodes = {0, 1};
    std::vector<uint8_t> output(4 * 3 * 2);
    std::vector<snark::FeatureMeta> features = {{0, 12}};

    g.GetNodeFeature(std::span(nodes), std::span(features), std::span(output));
    std::span res(reinterpret_cast<float *>(output.data()), output.size() / sizeof(float));
    EXPECT_EQ(std::vector<float>(std::begin(res), std::end(res)), std::vector<float>({1, 2, 3, 5, 6, 7}));
}

TEST_P(StorageTypeGraphTest, NodeFeaturesMultipleNodesSingleFeatureThreadPool)
{
    TestGraph::MemoryGraph m;
    std::vector<std::vector<float>> f1 = {std::vector<float>()};
    for (size_t i = 0; i < 1024; i++)
    {
        f1[0].push_back((float)i);
    }

    for (int64_t i = 0; i < 1024; i++)
    {
        m.m_nodes.push_back(TestGraph::Node{.m_id = i, .m_type = 0, .m_weight = 1.0f, .m_float_features = f1});
    }

    auto path = std::filesystem::temp_directory_path();
    TestGraph::convert(path, "0_0", std::move(m), 2);
    snark::Graph g(path.string(), std::vector<uint32_t>{0}, GetParam(), "");
    std::vector<snark::NodeId> nodes;
    for (size_t i = 0; i < 512; i++)
    {
        nodes.push_back(i);
    }

    std::vector<uint8_t> output(4 * 1024 * 512);
    std::vector<snark::FeatureMeta> features = {{0, 4096}};

    g.GetNodeFeature(std::span(nodes), std::span(features), std::span(output));
    std::span res(reinterpret_cast<float *>(output.data()), output.size() / sizeof(float));
    EXPECT_EQ(res.size(), 1024 * 512);
    EXPECT_EQ(std::vector<float>(std::begin(res), std::begin(res) + 5), std::vector<float>({0, 1, 2, 3, 4}));
}

TEST_P(StorageTypeGraphTest, NodeFeaturesMultipleNodesSingleFeatureMissingNode)
{
    TestGraph::MemoryGraph m;
    std::vector<std::vector<float>> f1 = {std::vector<float>{1.0f, 2.0f, 3.0f}};
    m.m_nodes.push_back(TestGraph::Node{.m_id = 0, .m_type = 0, .m_weight = 1.0f, .m_float_features = f1});
    auto path = std::filesystem::temp_directory_path();
    TestGraph::convert(path, "0_0", std::move(m), 2);
    snark::Graph g(path.string(), std::vector<uint32_t>{0}, GetParam(), "");
    std::vector<snark::NodeId> nodes = {0, 1};
    std::vector<uint8_t> output(4 * 3 * 2);
    std::vector<snark::FeatureMeta> features = {{0, 12}};

    g.GetNodeFeature(std::span(nodes), std::span(features), std::span(output));
    std::span res(reinterpret_cast<float *>(output.data()), output.size() / 4);
    EXPECT_EQ(std::vector<float>(std::begin(res), std::end(res)), std::vector<float>({1, 2, 3, 0, 0, 0}));
}

TEST_P(StorageTypeGraphTest, NodeFeaturesMultipleNodesMissingFeature)
{
    TestGraph::MemoryGraph m;
    std::vector<std::vector<float>> f1 = {std::vector<float>{1.0f, 2.0f, 3.0f}};
    std::vector<std::vector<float>> f2 = {std::vector<float>{11.0f, 12.0f}};
    m.m_nodes.push_back(TestGraph::Node{.m_id = 0, .m_type = 0, .m_weight = 1.0f, .m_float_features = f1});
    m.m_nodes.push_back(TestGraph::Node{.m_id = 1, .m_type = 1, .m_weight = 1.0f, .m_float_features = f2});
    m.m_nodes.push_back(TestGraph::Node{.m_id = 2, .m_type = 1, .m_weight = 1.0f, .m_float_features = f2});
    auto path = std::filesystem::temp_directory_path();
    TestGraph::convert(path, "0_0", std::move(m), 2);
    snark::Graph g(path.string(), std::vector<uint32_t>{0}, GetParam(), "");
    std::vector<snark::NodeId> nodes = {0};
    std::vector<uint8_t> large_output(4 * 3);
    std::vector<snark::FeatureMeta> features = {{1, 12}};

    g.GetNodeFeature(std::span(nodes), std::span(features), std::span(large_output));
    std::span large_res(reinterpret_cast<float *>(large_output.data()), large_output.size() / 4);
    EXPECT_EQ(std::vector<float>(std::begin(large_res), std::end(large_res)), std::vector<float>(3, 0.0f));
}

TEST_P(StorageTypeGraphTest, NodeFeaturesMultipleNodesSingleFeatureMixedSizes)
{
    TestGraph::MemoryGraph m;
    std::vector<std::vector<float>> f1 = {std::vector<float>{1.0f, 2.0f, 3.0f}};
    std::vector<std::vector<float>> f2 = {std::vector<float>{11.0f, 12.0f}};
    m.m_nodes.push_back(TestGraph::Node{.m_id = 0, .m_type = 0, .m_weight = 1.0f, .m_float_features = f1});
    m.m_nodes.push_back(TestGraph::Node{.m_id = 1, .m_type = 1, .m_weight = 1.0f, .m_float_features = f2});
    auto path = std::filesystem::temp_directory_path();
    TestGraph::convert(path, "0_0", std::move(m), 2);
    snark::Graph g(path.string(), std::vector<uint32_t>{0}, GetParam(), "");
    std::vector<snark::NodeId> nodes = {0, 1};
    std::vector<uint8_t> large_output(4 * 3 * 2);
    std::vector<snark::FeatureMeta> features = {{0, 12}};

    g.GetNodeFeature(std::span(nodes), std::span(features), std::span(large_output));
    std::span large_res(reinterpret_cast<float *>(large_output.data()), large_output.size() / 4);
    EXPECT_EQ(std::vector<float>(std::begin(large_res), std::end(large_res)), std::vector<float>({1, 2, 3, 11, 12, 0}));

    std::vector<uint8_t> short_output(4 * 2 * 2);

    features[0].second = 8;
    g.GetNodeFeature(std::span(nodes), std::span(features), std::span(short_output));
    std::span short_res(reinterpret_cast<float *>(short_output.data()), short_output.size() / 4);
    EXPECT_EQ(std::vector<float>(std::begin(short_res), std::end(short_res)), std::vector<float>({1, 2, 11, 12}));
}

TEST_P(StorageTypeGraphTest, NodeSparseFeaturesMultipleNodes)
{
    TestGraph::MemoryGraph m;
    // indices - 1, 13, 42, data - 1
    std::vector<int32_t> f1_data = {3, 3, 1, 0, 13, 0, 42, 0, 1};
    // indices - [3, 8, 9], [4, 3, 2] data - [5, 42]
    std::vector<int32_t> f2_data = {6, 3, 3, 0, 8, 0, 9, 0, 4, 0, 3, 0, 2, 0, 5, 42};
    auto start = reinterpret_cast<float *>(f1_data.data());
    std::vector<std::vector<float>> f1 = {std::vector<float>(start, start + f1_data.size())};
    start = reinterpret_cast<float *>(f2_data.data());
    std::vector<std::vector<float>> f2 = {std::vector<float>(start, start + f2_data.size())};
    m.m_nodes.push_back(TestGraph::Node{.m_id = 0, .m_type = 0, .m_weight = 1.0f, .m_float_features = f1});
    m.m_nodes.push_back(TestGraph::Node{.m_id = 1, .m_type = 1, .m_weight = 1.0f, .m_float_features = f2});
    auto path = std::filesystem::temp_directory_path();
    TestGraph::convert(path, "0_0", std::move(m), 2);
    snark::Graph g(path.string(), std::vector<uint32_t>{0}, GetParam(), "");
    std::vector<snark::NodeId> nodes = {0, 1};
    std::vector<snark::FeatureId> features = {0};

    std::vector<std::vector<uint8_t>> data(features.size());
    std::vector<std::vector<int64_t>> indices(features.size());
    std::vector<int64_t> dimensions = {-1};
    g.GetNodeSparseFeature(std::span(nodes), std::span(features), std::span(dimensions), indices, data);
    EXPECT_EQ(indices.size(), 1);
    EXPECT_EQ(data.size(), 1);
    EXPECT_EQ(std::vector<int64_t>({0, 1, 13, 42, 1, 3, 8, 9, 1, 4, 3, 2}), indices.front());
    auto tmp = reinterpret_cast<int32_t *>(data.front().data());
    EXPECT_EQ(std::vector<int32_t>({1, 5, 42}), std::vector<int32_t>(tmp, tmp + 3));
    EXPECT_EQ(std::vector<int64_t>({3}), dimensions);
}

TEST_P(StorageTypeGraphTest, NodeSparseFeaturesMixedDimensions)
{
    TestGraph::MemoryGraph m;
    // f_0: indices 2, data 4, f_1: indices - 1, 13, 42, data - 1
    std::vector<int32_t> f1_1_data = {1, 1, 2, 0, 4};
    std::vector<int32_t> f1_2_data = {3, 3, 1, 0, 13, 0, 42, 0, 1};
    // f_0: indices 6, data 5, f_1: indices - 3, 8, 9, data - 6
    std::vector<int32_t> f2_1_data = {1, 1, 6, 0, 5};
    std::vector<int32_t> f2_2_data = {3, 3, 3, 0, 8, 0, 9, 0, 6};
    auto start = reinterpret_cast<float *>(f1_1_data.data());
    std::vector<std::vector<float>> f1 = {std::vector<float>(start, start + f1_1_data.size())};
    start = reinterpret_cast<float *>(f1_2_data.data());
    f1.emplace_back(start, start + f1_2_data.size());
    start = reinterpret_cast<float *>(f2_1_data.data());
    std::vector<std::vector<float>> f2 = {std::vector<float>(start, start + f2_1_data.size())};
    start = reinterpret_cast<float *>(f2_2_data.data());
    f2.emplace_back(start, start + f2_2_data.size());
    m.m_nodes.push_back(TestGraph::Node{.m_id = 0, .m_type = 0, .m_weight = 1.0f, .m_float_features = f1});
    m.m_nodes.push_back(TestGraph::Node{.m_id = 1, .m_type = 1, .m_weight = 1.0f, .m_float_features = f2});
    auto path = std::filesystem::temp_directory_path();
    TestGraph::convert(path, "0_0", std::move(m), 2);
    snark::Graph g(path.string(), std::vector<uint32_t>{0}, GetParam(), "");
    std::vector<snark::NodeId> nodes = {0, 1};
    std::vector<snark::FeatureId> features = {0, 1};

    std::vector<std::vector<uint8_t>> data(features.size());
    std::vector<std::vector<int64_t>> indices(features.size());
    std::vector<int64_t> dimensions = {-1, -1};
    g.GetNodeSparseFeature(std::span(nodes), std::span(features), std::span(dimensions), indices, data);
    EXPECT_EQ(indices.size(), 2);
    EXPECT_EQ(data.size(), 2);
    EXPECT_EQ(std::vector<int64_t>({0, 2, 1, 6}), indices[0]);
    EXPECT_EQ(std::vector<int64_t>({0, 1, 13, 42, 1, 3, 8, 9}), indices[1]);
    EXPECT_EQ(std::vector<int64_t>({1, 3}), dimensions);
    auto tmp = reinterpret_cast<int32_t *>(data[0].data());
    EXPECT_EQ(std::vector<int32_t>({4, 5}), std::vector<int32_t>(tmp, tmp + 2));
    tmp = reinterpret_cast<int32_t *>(data[1].data());
    EXPECT_EQ(std::vector<int32_t>({1, 6}), std::vector<int32_t>(tmp, tmp + 2));
}

TEST_P(StorageTypeGraphTest, NodeSparseFeaturesMissingFeature)
{
    TestGraph::MemoryGraph m;
    // indices - 1, 13, 42, data - 7
    std::vector<int32_t> feature_data = {3, 3, 1, 0, 13, 0, 42, 0, 7};
    auto start = reinterpret_cast<float *>(feature_data.data());
    std::vector<std::vector<float>> f1 = {std::vector<float>(start, start + feature_data.size())};
    m.m_nodes.push_back(TestGraph::Node{.m_id = 0, .m_type = 0, .m_weight = 1.0f, .m_float_features = f1});
    m.m_nodes.push_back(TestGraph::Node{.m_id = 1, .m_type = 1, .m_weight = 1.0f});
    auto path = std::filesystem::temp_directory_path();
    TestGraph::convert(path, "0_0", std::move(m), 2);
    snark::Graph g(path.string(), std::vector<uint32_t>{0}, GetParam(), "");
    std::vector<snark::NodeId> nodes = {0, 1};
    std::vector<snark::FeatureId> features = {0};

    std::vector<std::vector<uint8_t>> data(features.size());
    std::vector<std::vector<int64_t>> indices(features.size());
    std::vector<int64_t> dimensions = {-1};
    g.GetNodeSparseFeature(std::span(nodes), std::span(features), std::span(dimensions), indices, data);
    EXPECT_EQ(std::vector<int64_t>({0, 1, 13, 42}), indices.front());
    auto tmp = reinterpret_cast<int32_t *>(data.front().data());
    EXPECT_EQ(std::vector<int32_t>({7}), std::vector<int32_t>(tmp, tmp + 1));
    EXPECT_EQ(std::vector<int64_t>({3}), dimensions);
}

TEST_P(StorageTypeGraphTest, NodeSparseFeaturesDimensionsFill)
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
    auto path = std::filesystem::temp_directory_path();
    auto partition = TestGraph::convert(path, "0_0", std::move(m), 1);
    snark::Graph g(path.string(), std::vector<uint32_t>{0}, GetParam(), "");

    std::vector<snark::NodeId> nodes = {13979298};
    std::vector<snark::FeatureId> features = {6};

    std::vector<std::vector<uint8_t>> data(features.size());
    std::vector<std::vector<int64_t>> indices(features.size());
    std::vector<int64_t> dimensions = {-1};
    g.GetNodeSparseFeature(std::span(nodes), std::span(features), std::span(dimensions), indices, data);
    EXPECT_EQ(std::vector<int64_t>({0, 0}), indices[0]);
    EXPECT_EQ(std::vector<int64_t>({1}), dimensions);
    auto tmp = reinterpret_cast<float *>(data[0].data());
    EXPECT_EQ(std::vector<float>({1.0}), std::vector<float>(tmp, tmp + 1));

    features = {1, 6};

    data = {{}, {}};
    indices = {{}, {}};
    dimensions = {-1, -1};
    g.GetNodeSparseFeature(std::span(nodes), std::span(features), std::span(dimensions), indices, data);
    EXPECT_EQ(std::vector<int64_t>({}), indices[0]);
    EXPECT_EQ(std::vector<int64_t>({0, 0}), indices[1]);
    EXPECT_EQ(std::vector<int64_t>({0, 1}), dimensions);
    tmp = reinterpret_cast<float *>(data[1].data());
    EXPECT_EQ(std::vector<float>({1.0}), std::vector<float>(tmp, tmp + 1));

    features = {1, 2, 5, 6};
    data = {{}, {}, {}, {}};
    indices = {{}, {}, {}, {}};
    dimensions = {-1, -1, -1, -1};
    g.GetNodeSparseFeature(std::span(nodes), std::span(features), std::span(dimensions), indices, data);
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
    g.GetNodeSparseFeature(std::span(nodes), std::span(features), std::span(dimensions), indices, data);
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

TEST_P(StorageTypeGraphTest, NodeStringFeaturesMultipleNodesSingleFeature)
{
    TestGraph::MemoryGraph m;
    std::vector<std::vector<float>> f1 = {std::vector<float>{11.0f, 12.0f}};
    std::vector<std::vector<float>> f2 = {std::vector<float>{1.0f, 2.0f, 3.0f}};
    m.m_nodes.push_back(TestGraph::Node{.m_id = 0, .m_type = 0, .m_weight = 1.0f, .m_float_features = f1});
    m.m_nodes.push_back(TestGraph::Node{.m_id = 1, .m_type = 1, .m_weight = 1.0f});
    m.m_nodes.push_back(TestGraph::Node{.m_id = 2, .m_type = 1, .m_weight = 1.0f, .m_float_features = f2});
    auto path = std::filesystem::temp_directory_path();
    TestGraph::convert(path, "0_0", std::move(m), 2);
    snark::Graph g(path.string(), std::vector<uint32_t>{0}, GetParam(), "");
    std::vector<snark::NodeId> nodes = {0, 1, 2};
    std::vector<uint8_t> output;
    std::vector<int64_t> dimensions(3);
    std::vector<snark::FeatureId> features = {0};

    g.GetNodeStringFeature(std::span(nodes), std::span(features), std::span(dimensions), output);
    std::span res(reinterpret_cast<float *>(output.data()), output.size() / 4);
    EXPECT_EQ(std::vector<float>(std::begin(res), std::end(res)), std::vector<float>({11, 12, 1, 2, 3}));
    EXPECT_EQ(dimensions, std::vector<int64_t>({8, 0, 12}));
}

TEST_P(StorageTypeGraphTest, NeighborSamplesWithSingleNodeNoNeighbors)
{
    TestGraph::MemoryGraph m;
    m.m_nodes.push_back(TestGraph::Node{.m_id = 0, .m_type = 0, .m_weight = 1.0f});
    m.m_nodes.push_back(TestGraph::Node{.m_id = 1, .m_type = 1, .m_weight = 1.0f});
    auto path = std::filesystem::temp_directory_path();
    TestGraph::convert(path, "0_0", std::move(m), 2);
    snark::Graph g(path.string(), std::vector<uint32_t>{0}, GetParam(), "");
    std::vector<snark::NodeId> nodes = {0, 1};
    std::vector<snark::Type> types = {0};
    int count = 5;
    std::vector<snark::NodeId> neighbor_nodes(count * nodes.size(), -1);
    std::vector<snark::Type> neighbor_types(count * nodes.size(), -1);
    std::vector<float> neighbor_weights(count * nodes.size(), -1);
    std::vector<float> total_neighbor_weights(nodes.size());

    g.SampleNeighbor(42, std::span(nodes), std::span(types), count, std::span(neighbor_nodes),
                     std::span(neighbor_types), std::span(neighbor_weights), std::span(total_neighbor_weights), 13, 2,
                     -1);
    EXPECT_EQ(std::vector<snark::NodeId>(10, 13), neighbor_nodes);
    EXPECT_EQ(std::vector<snark::Type>(10, -1), neighbor_types);
    EXPECT_EQ(std::vector<float>(10, 2.0f), neighbor_weights);
}

TEST(GraphTest, NeighborSampleSimple)
{
    TestGraph::MemoryGraph m;
    m.m_nodes.push_back(TestGraph::Node{
        .m_id = 0,
        .m_type = 0,
        .m_weight = 1.0f,
        .m_neighbors{std::vector<TestGraph::NeighborRecord>{{1, 0, 1.0f}, {2, 0, 1.0f}, {3, 0, 1.0f}, {4, 0, 1.0f}}}});
    m.m_nodes.push_back(TestGraph::Node{
        .m_id = 2,
        .m_type = 1,
        .m_weight = 1.0f,
        .m_neighbors{std::vector<TestGraph::NeighborRecord>{{5, 0, 1.0f}, {6, 0, 1.0f}, {7, 0, 1.0f}, {8, 0, 1.0f}}}});

    auto path = std::filesystem::temp_directory_path();
    TestGraph::convert(path, "0_0", std::move(m), 2);
    snark::Graph g(path.string(), {0}, snark::PartitionStorageType::memory, "");
    std::vector<snark::NodeId> nodes = {0, 2};
    std::vector<snark::Type> types = {0};
    int count = 3;
    std::vector<snark::NodeId> neighbor_nodes(count * nodes.size(), -1);
    std::vector<snark::Type> neighbor_types(count * nodes.size(), -1);
    std::vector<float> neighbor_weights(count * nodes.size(), -1);
    std::vector<float> total_neighbor_weights(nodes.size());

    g.SampleNeighbor(42, std::span(nodes), std::span(types), count, std::span(neighbor_nodes),
                     std::span(neighbor_types), std::span(neighbor_weights), std::span(total_neighbor_weights), 0, 0,
                     -1);
    EXPECT_EQ(std::vector<snark::NodeId>({4, 1, 3, 6, 6, 8}), neighbor_nodes);
    EXPECT_EQ(std::vector<snark::Type>(6, 0), neighbor_types);
    EXPECT_EQ(std::vector<float>(6, 1), neighbor_weights);
}

TEST(GraphTest, NeighborSampleSimpleThreadPool)
{
    TestGraph::MemoryGraph m;

    for (int64_t i = 0; i < 1024; i++)
    {
        std::vector<TestGraph::NeighborRecord> neighbors;
        for (int64_t k = 0; k < 64; k++)
        {
            neighbors.push_back(TestGraph::NeighborRecord{i + k, 0, 1.0f});
        }

        m.m_nodes.push_back(TestGraph::Node{.m_id = i, .m_type = 0, .m_weight = 1.0f, .m_neighbors = neighbors});
    }

    auto path = std::filesystem::temp_directory_path();
    TestGraph::convert(path, "0_0", std::move(m), 2);
    snark::Graph g(path.string(), {0}, snark::PartitionStorageType::memory, "");
    std::vector<snark::NodeId> nodes;
    for (int64_t i = 0; i < 256; i++)
    {
        nodes.push_back(i);
    }
    std::vector<snark::Type> types = {0};
    int count = 64;
    std::vector<snark::NodeId> neighbor_nodes(count * nodes.size(), -1);
    std::vector<snark::Type> neighbor_types(count * nodes.size(), -1);
    std::vector<float> neighbor_weights(count * nodes.size(), -1);
    std::vector<float> total_neighbor_weights(nodes.size());

    g.SampleNeighbor(42, std::span(nodes), std::span(types), count, std::span(neighbor_nodes),
                     std::span(neighbor_types), std::span(neighbor_weights), std::span(total_neighbor_weights), 0, 0,
                     -1);
    EXPECT_EQ(std::vector<snark::NodeId>({57, 4, 33, 20, 30}),
              std::vector<snark::NodeId>(std::begin(neighbor_nodes), std::begin(neighbor_nodes) + 5));
    EXPECT_EQ(std::vector<snark::Type>(16384, 0), neighbor_types);
    EXPECT_EQ(std::vector<float>(16384, 1), neighbor_weights);
}

TEST_P(StorageTypeGraphTest, NeighborSampleMultipleTypesSinglePartition)
{
    TestGraph::MemoryGraph m;
    m.m_nodes.push_back(TestGraph::Node{.m_id = 0,
                                        .m_type = 0,
                                        .m_weight = 1.0f,
                                        .m_neighbors{std::vector<TestGraph::NeighborRecord>{
                                            {1, 0, 1.0f}, {2, 0, 1.0f}, {3, 0, 1.0f}, {4, 0, 1.0f}, {5, 0, 1.0f}}}});
    m.m_nodes.push_back(TestGraph::Node{.m_id = 1, .m_type = 1, .m_weight = 1.0f});
    m.m_nodes.push_back(TestGraph::Node{
        .m_id = 2,
        .m_type = 1,
        .m_weight = 1.0f,
        .m_neighbors{std::vector<TestGraph::NeighborRecord>{{6, 0, 1.0f}, {7, 0, 1.0f}, {8, 0, 1.0f}}}});
    auto path = std::filesystem::temp_directory_path();
    TestGraph::convert(path, "0_0", std::move(m), 2);
    snark::Graph g(path.string(), std::vector<uint32_t>{0}, GetParam(), "");
    std::vector<snark::NodeId> nodes = {0, 2};
    std::vector<snark::Type> types = {0, 1};
    int count = 3;
    std::vector<snark::NodeId> neighbor_nodes(count * nodes.size(), -1);
    std::vector<snark::Type> neighbor_types(count * nodes.size(), -1);
    std::vector<float> neighbor_weights(count * nodes.size(), -1);
    std::vector<float> total_neighbor_weights(nodes.size());

    g.SampleNeighbor(42, std::span(nodes), std::span(types), count, std::span(neighbor_nodes),
                     std::span(neighbor_types), std::span(neighbor_weights), std::span(total_neighbor_weights), 0, 0,
                     -1);

    EXPECT_EQ(std::vector<snark::NodeId>({5, 1, 3, 7, 7, 8}), neighbor_nodes);
    EXPECT_EQ(std::vector<snark::Type>(6, 0), neighbor_types);
    EXPECT_EQ(std::vector<float>(6, 1.f), neighbor_weights);
}

TEST(GraphTest, NeighborSampleMultipleTypesMultiplePartitions)
{
    TestGraph::MemoryGraph m1;
    m1.m_nodes.push_back(
        TestGraph::Node{.m_id = 0,
                        .m_type = 0,
                        .m_weight = 1.0f,
                        .m_float_features = {},
                        .m_neighbors{std::vector<TestGraph::NeighborRecord>{{1, 0, 1.0f}, {2, 0, 1.0f}}}});
    m1.m_nodes.push_back(TestGraph::Node{.m_id = 1, .m_type = 1, .m_weight = 1.0f});
    TestGraph::MemoryGraph m2;
    m2.m_nodes.push_back(TestGraph::Node{
        .m_id = 2,
        .m_type = -1,
        .m_weight = 1.0f,
        .m_neighbors{std::vector<TestGraph::NeighborRecord>{{3, 0, 1.0f}, {4, 0, 3.0f}, {5, 1, 0.5f}, {6, 1, 2.0f}}}});
    auto path = std::filesystem::temp_directory_path();
    TestGraph::convert(path, "0_0", std::move(m1), 3);
    TestGraph::convert(path, "1_0", std::move(m2), 3);
    snark::Graph g(path.string(), {0, 1}, snark::PartitionStorageType::memory, "");
    std::vector<snark::NodeId> nodes = {0, 2};
    std::vector<snark::Type> types = {0, 1};
    int count = 2;
    std::vector<snark::NodeId> neighbor_nodes(count * nodes.size(), -1);
    std::vector<snark::Type> neighbor_types(count * nodes.size(), -1);
    std::vector<float> neighbor_weights(count * nodes.size(), -1);
    std::vector<float> total_neighbor_weights(nodes.size());

    g.SampleNeighbor(8, std::span(nodes), std::span(types), count, std::span(neighbor_nodes), std::span(neighbor_types),
                     std::span(neighbor_weights), std::span(total_neighbor_weights), 0, 0, 0);
    EXPECT_EQ(std::vector<snark::NodeId>({1, 1, 4, 6}), neighbor_nodes);
    EXPECT_EQ(std::vector<snark::Type>({0, 0, 0, 1}), neighbor_types);
    EXPECT_EQ(std::vector<float>({1.f, 1.f, 3.f, 2.0f}), neighbor_weights);
}

TEST(GraphTest, NeighborSampleMultipleTypesNeighborsSpreadAcrossPartitions)
{
    TestGraph::MemoryGraph m1;
    m1.m_nodes.push_back(
        TestGraph::Node{.m_id = 0,
                        .m_type = 0,
                        .m_weight = 1.0f,
                        .m_neighbors{std::vector<TestGraph::NeighborRecord>{{1, 0, 1.0f}, {2, 0, 1.0f}}}});
    m1.m_nodes.push_back(TestGraph::Node{
        .m_id = 1,
        .m_type = -1,
        .m_weight = 1.0f,
        .m_neighbors{std::vector<TestGraph::NeighborRecord>{{3, 0, 1.0f}, {4, 0, 1.0f}, {5, 1, 1.0f}}}});
    TestGraph::MemoryGraph m2;
    m2.m_nodes.push_back(TestGraph::Node{
        .m_id = 1, .m_type = 1, .m_neighbors{std::vector<TestGraph::NeighborRecord>{{6, 1, 1.5f}, {7, 1, 3.0f}}}});
    auto path = std::filesystem::temp_directory_path();

    TestGraph::convert(path, "0_0", std::move(m1), 2);
    TestGraph::convert(path, "1_0", std::move(m2), 2);
    snark::Graph g(path.string(), {0, 1}, snark::PartitionStorageType::memory, "");
    std::vector<snark::NodeId> nodes = {1};
    std::vector<snark::Type> types = {0, 1};
    int count = 6;
    std::vector<snark::NodeId> neighbor_nodes(count * nodes.size(), -1);
    std::vector<snark::Type> neighbor_types(count * nodes.size(), -1);
    std::vector<float> neighbor_weights(count * nodes.size(), -1);
    std::vector<float> total_neighbor_weights(nodes.size());

    g.SampleNeighbor(13, std::span(nodes), std::span(types), count, std::span(neighbor_nodes),
                     std::span(neighbor_types), std::span(neighbor_weights), std::span(total_neighbor_weights), 0, 0,
                     0);

    EXPECT_EQ(std::vector<snark::NodeId>({7, 7, 7, 7, 3, 5}), neighbor_nodes);
    EXPECT_EQ(std::vector<snark::Type>({1, 1, 1, 1, 0, 1}), neighbor_types);
    EXPECT_EQ(std::vector<float>({3.f, 3.f, 3.f, 3.f, 1.f, 1.f}), neighbor_weights);
}

TEST(GraphTest, StatisticalNeighborSampleMultipleTypesNeighborsSpreadAcrossPartitions)
{
    TestGraph::MemoryGraph m1;
    m1.m_nodes.push_back(
        TestGraph::Node{.m_id = 0,
                        .m_type = 0,
                        .m_weight = 1.0f,
                        .m_neighbors{std::vector<TestGraph::NeighborRecord>{{1, 0, 1.0f}, {2, 0, 1.0f}}}});
    m1.m_nodes.push_back(TestGraph::Node{
        .m_id = 1,
        .m_type = 1,
        .m_weight = 1.0f,
        .m_neighbors{std::vector<TestGraph::NeighborRecord>{{3, 0, 1.0f}, {4, 0, 1.0f}, {5, 1, 1.0f}}}});
    TestGraph::MemoryGraph m2;
    m2.m_nodes.push_back(TestGraph::Node{
        .m_id = 1, .m_type = -1, .m_neighbors{std::vector<TestGraph::NeighborRecord>{{6, 1, 1.0f}, {7, 1, 1.0f}}}});
    auto path = std::filesystem::temp_directory_path();
    TestGraph::convert(path, "0_0", std::move(m1), 2);
    TestGraph::convert(path, "1_0", std::move(m2), 2);
    snark::Graph g(path.string(), {0, 1}, snark::PartitionStorageType::memory, "");
    std::vector<snark::NodeId> nodes = {1};
    std::vector<snark::Type> types = {1};
    std::vector<size_t> sample_counts(9);
    int count = 3;
    std::vector<snark::NodeId> neighbor_nodes(count * nodes.size(), -1);
    std::vector<snark::Type> neighbor_types(count, -1);
    std::vector<float> neighbor_weights(count * nodes.size(), -1);
    snark::Xoroshiro128PlusGenerator gen(42);
    const size_t repetitions = 10000;
    boost::random::uniform_int_distribution<int64_t> seeds;
    std::vector<float> total_neighbor_weights(nodes.size());
    for (size_t i = 0; i < repetitions; ++i)
    {
        std::fill(std::begin(total_neighbor_weights), std::end(total_neighbor_weights), 0);
        g.SampleNeighbor(seeds(gen), std::span(nodes), std::span(types), count, std::span(neighbor_nodes),
                         std::span(neighbor_types), std::span(neighbor_weights), std::span(total_neighbor_weights), 0,
                         0, 0);
        for (auto n : neighbor_nodes)
        {
            if (n < 0)
                continue;
            ++sample_counts[n];
        }
    }

    EXPECT_EQ(std::vector<size_t>({0, 0, 0, 0, 0, 9965, 9908, 10127, 0}), sample_counts);
}

TEST(GraphTest, UniformNeighborSampleMultipleTypesNeighborsSpreadAcrossPartitions)
{
    TestGraph::MemoryGraph m1;
    m1.m_nodes.push_back(
        TestGraph::Node{.m_id = 0,
                        .m_type = 0,
                        .m_weight = 1.0f,
                        .m_neighbors{std::vector<TestGraph::NeighborRecord>{{1, 0, 1.0f}, {2, 0, 1.0f}}}});
    m1.m_nodes.push_back(TestGraph::Node{
        .m_id = 1,
        .m_type = 1,
        .m_weight = 1.0f,
        .m_neighbors{std::vector<TestGraph::NeighborRecord>{{3, 0, 1.0f}, {4, 0, 1.0f}, {5, 1, 1.0f}}}});
    TestGraph::MemoryGraph m2;
    m2.m_nodes.push_back(TestGraph::Node{
        .m_id = 1, .m_type = -1, .m_neighbors{std::vector<TestGraph::NeighborRecord>{{6, 1, 1.5f}, {7, 1, 3.0f}}}});
    auto path = std::filesystem::temp_directory_path();
    TestGraph::convert(path, "0_0", std::move(m1), 2);
    TestGraph::convert(path, "1_0", std::move(m2), 2);
    snark::Graph g(path.string(), {0, 1}, snark::PartitionStorageType::memory, "");
    std::vector<snark::NodeId> nodes = {1};
    std::vector<snark::Type> types = {0, 1};
    int count = 6;
    std::vector<snark::NodeId> neighbor_nodes(count * nodes.size(), -1);
    std::vector<snark::Type> neighbor_types(count * nodes.size(), -1);
    std::vector<uint64_t> total_neighbor_counts(nodes.size());

    g.UniformSampleNeighbor(true, 17, std::span(nodes), std::span(types), count, std::span(neighbor_nodes),
                            std::span(neighbor_types), std::span(total_neighbor_counts), 0, 2);
    EXPECT_EQ(std::vector<snark::NodeId>({6, 5, 4, 7, 3, 0}), neighbor_nodes);
    EXPECT_EQ(std::vector<snark::Type>({1, 1, 0, 1, 0, 2}), neighbor_types);
}

TEST(GraphTest, UniformNeighborSampleNeighborsThreadPool)
{
    TestGraph::MemoryGraph m1;
    for (int64_t i = 0; i < 1024; i++)
    {
        std::vector<TestGraph::NeighborRecord> neighbors;
        for (int64_t k = 0; k < 64; k++)
        {
            neighbors.push_back(TestGraph::NeighborRecord{i + k, 0, 1.0f});
        }

        m1.m_nodes.push_back(TestGraph::Node{.m_id = i, .m_type = 0, .m_weight = 1.0f, .m_neighbors = neighbors});
    }

    auto path = std::filesystem::temp_directory_path();
    TestGraph::convert(path, "0_0", std::move(m1), 2);
    snark::Graph g(path.string(), {0}, snark::PartitionStorageType::memory, "");
    std::vector<snark::NodeId> nodes;
    for (int64_t i = 0; i < 256; i++)
    {
        nodes.push_back(i);
    }

    std::vector<snark::Type> types = {0};
    int count = 64;
    std::vector<snark::NodeId> neighbor_nodes(count * nodes.size(), -1);
    std::vector<snark::Type> neighbor_types(count * nodes.size(), -1);
    std::vector<uint64_t> total_neighbor_counts(nodes.size());

    g.UniformSampleNeighbor(true, 17, std::span(nodes), std::span(types), count, std::span(neighbor_nodes),
                            std::span(neighbor_types), std::span(total_neighbor_counts), 0, 2);
    EXPECT_EQ(std::vector<snark::NodeId>({57, 56, 15, 35, 4}),
              std::vector<snark::NodeId>(std::begin(neighbor_nodes), std::begin(neighbor_nodes) + 5));
}

TEST(GraphTest, NodeTypesMultipleTypesNeighborsSpreadAcrossPartitions)
{
    TestGraph::MemoryGraph m1;
    m1.m_nodes.push_back(
        TestGraph::Node{.m_id = 0,
                        .m_type = 0,
                        .m_weight = 1.0f,
                        .m_neighbors{std::vector<TestGraph::NeighborRecord>{{1, 0, 1.0f}, {2, 0, 1.0f}}}});
    m1.m_nodes.push_back(TestGraph::Node{
        .m_id = 1,
        .m_type = 1,
        .m_weight = 1.0f,
        .m_neighbors{std::vector<TestGraph::NeighborRecord>{{3, 0, 1.0f}, {4, 0, 1.0f}, {5, 1, 1.0f}}}});
    m1.m_nodes.push_back(TestGraph::Node{
        .m_id = 2,
        .m_type = -1,
        .m_weight = 1.0f,
        .m_neighbors{std::vector<TestGraph::NeighborRecord>{{3, 0, 1.0f}, {4, 0, 1.0f}, {5, 1, 1.0f}}}});
    TestGraph::MemoryGraph m2;
    m2.m_nodes.push_back(TestGraph::Node{
        .m_id = 1, .m_type = -1, .m_neighbors{std::vector<TestGraph::NeighborRecord>{{6, 1, 1.5f}, {7, 1, 3.0f}}}});
    m2.m_nodes.push_back(TestGraph::Node{
        .m_id = 2, .m_type = 2, .m_neighbors{std::vector<TestGraph::NeighborRecord>{{6, 1, 1.5f}, {7, 1, 3.0f}}}});
    auto path = std::filesystem::temp_directory_path();

    TestGraph::convert(path, "0_0", std::move(m1), 3);
    TestGraph::convert(path, "1_0", std::move(m2), 3);
    snark::Graph g(path.string(), {0, 1}, snark::PartitionStorageType::memory, "");

    std::vector<snark::NodeId> nodes = {0, 1, 2};
    std::vector<snark::Type> types(3, -3);

    g.GetNodeType(std::span(nodes), std::span(types), -2);
    EXPECT_EQ(std::vector<snark::Type>({0, 1, 2}), types);
}

TEST(GraphTest, NodeFeaturesMultipleTypesNeighborsSpreadAcrossPartitions)
{
    TestGraph::MemoryGraph m1;
    std::vector<std::vector<float>> f0 = {std::vector<float>{1.0f, 2.0f, 3.0f}};
    std::vector<std::vector<float>> f1 = {std::vector<float>{4.0f, 5.0f, 6.0f}};
    std::vector<std::vector<float>> f2 = {std::vector<float>{7.0f, 8.0f, 9.0f}};

    m1.m_nodes.push_back(TestGraph::Node{.m_id = 0, .m_type = 0, .m_weight = 1.0f, .m_float_features = f0});
    m1.m_nodes.push_back(TestGraph::Node{.m_id = 1, .m_type = 1, .m_weight = 1.0f, .m_float_features = f1});
    m1.m_nodes.push_back(TestGraph::Node{.m_id = 2, .m_type = -1, .m_weight = 1.0f});
    TestGraph::MemoryGraph m2;
    m2.m_nodes.push_back(TestGraph::Node{.m_id = 1, .m_type = -1});
    m2.m_nodes.push_back(TestGraph::Node{.m_id = 2, .m_type = 2, .m_float_features = f2});
    auto path = std::filesystem::temp_directory_path();

    TestGraph::convert(path, "0_0", std::move(m1), 3);
    TestGraph::convert(path, "1_0", std::move(m2), 3);
    snark::Graph g(path.string(), {0, 1}, snark::PartitionStorageType::memory, "");

    // 0 is a normal node
    // 1, 2 has a parity with type = -1
    // 3 is non existant
    std::vector<snark::NodeId> nodes = {0, 1, 2, 3};
    std::vector<uint8_t> output(4 * 3 * 4);
    std::vector<snark::FeatureMeta> features = {{0, 12}};

    g.GetNodeFeature(std::span(nodes), std::span(features), std::span(output));
    std::span res(reinterpret_cast<float *>(output.data()), output.size() / sizeof(float));
    EXPECT_EQ(std::vector<float>(std::begin(res), std::end(res)),
              std::vector<float>({1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 0, 0}));
}

TEST(GraphTest, NodeStringFeaturesMultipleTypesNeighborsSpreadAcrossPartitions)
{
    TestGraph::MemoryGraph m1;
    std::vector<std::vector<float>> f0 = {std::vector<float>{1.0f, 2.0f, 3.0f}};
    std::vector<std::vector<float>> f1 = {std::vector<float>{4.0f, 5.0f, 6.0f}};
    std::vector<std::vector<float>> f2 = {std::vector<float>{7.0f, 8.0f, 9.0f}};

    m1.m_nodes.push_back(TestGraph::Node{.m_id = 0, .m_type = 0, .m_weight = 1.0f, .m_float_features = f0});
    m1.m_nodes.push_back(TestGraph::Node{.m_id = 1, .m_type = 1, .m_weight = 1.0f, .m_float_features = f1});
    m1.m_nodes.push_back(TestGraph::Node{.m_id = 2, .m_type = -1, .m_weight = 1.0f});
    TestGraph::MemoryGraph m2;
    m2.m_nodes.push_back(TestGraph::Node{.m_id = 1, .m_type = -1});
    m2.m_nodes.push_back(TestGraph::Node{.m_id = 2, .m_type = 2, .m_float_features = f2});
    auto path = std::filesystem::temp_directory_path();

    TestGraph::convert(path, "0_0", std::move(m1), 3);
    TestGraph::convert(path, "1_0", std::move(m2), 3);
    snark::Graph g(path.string(), {0, 1}, snark::PartitionStorageType::memory, "");

    // 0 is a normal node
    // 1, 2 has a parity with type = -1
    // 3 is non existant
    std::vector<snark::NodeId> nodes = {0, 1, 2, 3};
    std::vector<uint8_t> output;
    std::vector<int64_t> dimensions(4);
    std::vector<snark::FeatureId> features = {0};
    g.GetNodeStringFeature(std::span(nodes), std::span(features), std::span(dimensions), output);
    std::span res(reinterpret_cast<float *>(output.data()), output.size() / sizeof(float));
    EXPECT_EQ(std::vector<float>(std::begin(res), std::end(res)), std::vector<float>({1, 2, 3, 4, 5, 6, 7, 8, 9}));
    EXPECT_EQ(dimensions, std::vector<int64_t>({12, 12, 12, 0}));
}

TEST(GraphTest, NodeSparseFeaturesMultipleTypesNeighborsSpreadAcrossPartitions)
{
    TestGraph::MemoryGraph m1;
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

    m1.m_nodes.push_back(TestGraph::Node{.m_id = 0, .m_type = 0, .m_weight = 1.0f, .m_float_features = f0});
    m1.m_nodes.push_back(TestGraph::Node{.m_id = 1, .m_type = 1, .m_weight = 1.0f, .m_float_features = f1});
    m1.m_nodes.push_back(TestGraph::Node{.m_id = 2, .m_type = -1, .m_weight = 1.0f});
    TestGraph::MemoryGraph m2;
    m2.m_nodes.push_back(TestGraph::Node{.m_id = 1, .m_type = -1});
    m2.m_nodes.push_back(TestGraph::Node{.m_id = 2, .m_type = 2, .m_weight = 1.0f, .m_float_features = f2});
    auto path = std::filesystem::temp_directory_path();

    TestGraph::convert(path, "0_0", std::move(m1), 3);
    TestGraph::convert(path, "1_0", std::move(m2), 3);
    snark::Graph g(path.string(), {0, 1}, snark::PartitionStorageType::memory, "");

    // 0 is a normal node
    // 1, 2 has a parity with type = -1
    // 3 is non existant
    std::vector<snark::NodeId> nodes = {0, 1, 2, 3};
    std::vector<snark::FeatureId> features = {0};
    std::vector<std::vector<uint8_t>> data(features.size());
    std::vector<std::vector<int64_t>> indices(features.size());
    std::vector<int64_t> dimensions = {-1};

    g.GetNodeSparseFeature(std::span(nodes), std::span(features), std::span(dimensions), indices, data);
    EXPECT_EQ(indices.size(), 1);
    EXPECT_EQ(data.size(), 1);
    EXPECT_EQ(std::vector<int64_t>({0, 1, 14, 20, 1, 1, 13, 42, 2, 3, 8, 9, 2, 4, 3, 2}), indices.front());
    auto tmp = reinterpret_cast<int32_t *>(data.front().data());
    EXPECT_EQ(std::vector<int32_t>({1, 1, 5, 42}), std::vector<int32_t>(tmp, tmp + 4));
    EXPECT_EQ(std::vector<int64_t>({3}), dimensions);
}

TEST(GraphTest, UniformNeighborSampleMultipleTypesTriggerConditionalProbabilities)
{
    TestGraph::MemoryGraph m1;
    m1.m_nodes.push_back(TestGraph::Node{.m_id = 0,
                                         .m_type = 0,
                                         .m_weight = 1.0f,
                                         .m_neighbors{std::vector<TestGraph::NeighborRecord>{{1, 0, 1.0f},
                                                                                             {2, 0, 1.0f},
                                                                                             {3, 1, 1.0f},
                                                                                             {4, 1, 1.0f},
                                                                                             {5, 2, 1.0f},
                                                                                             {6, 2, 1.0f},
                                                                                             {7, 3, 1.0f},
                                                                                             {8, 3, 1.0f}}}});
    m1.m_nodes.push_back(TestGraph::Node{
        .m_id = 1,
        .m_type = 0,
        .m_weight = 1.0f,
        .m_neighbors{std::vector<TestGraph::NeighborRecord>{{3, 0, 1.0f}, {4, 0, 1.0f}, {5, 1, 1.0f}}}});
    auto path = std::filesystem::temp_directory_path();
    TestGraph::convert(path, "0_0", std::move(m1), 1);
    snark::Graph g(path.string(), {0}, snark::PartitionStorageType::memory, "");
    std::vector<snark::NodeId> nodes = {0};
    std::vector<snark::Type> types = {0, 1, 3};
    int count = 6;
    std::vector<snark::NodeId> neighbor_nodes(count * nodes.size(), -1);
    std::vector<snark::Type> neighbor_types(count * nodes.size(), -1);
    std::vector<uint64_t> total_neighbor_counts(nodes.size());

    g.UniformSampleNeighbor(false, 3, std::span(nodes), std::span(types), count, std::span(neighbor_nodes),
                            std::span(neighbor_types), std::span(total_neighbor_counts), 0, 0);
    EXPECT_EQ(std::vector<snark::NodeId>({2, 3, 3, 7, 4, 2}), neighbor_nodes);
    EXPECT_EQ(std::vector<snark::Type>({0, 1, 1, 3, 1, 0}), neighbor_types);
}

TEST(GraphTest, StatisticalUniformNeighborSampleSingleTypeNeighborSpreadAcrossPartitions)
{
    TestGraph::MemoryGraph m1;
    m1.m_nodes.push_back(
        TestGraph::Node{.m_id = 0,
                        .m_type = 0,
                        .m_weight = 1.0f,
                        .m_neighbors{std::vector<TestGraph::NeighborRecord>{{1, 0, 1.0f}, {2, 0, 1.0f}}}});
    m1.m_nodes.push_back(TestGraph::Node{
        .m_id = 1,
        .m_type = 1,
        .m_weight = 1.0f,
        .m_neighbors{std::vector<TestGraph::NeighborRecord>{{3, 0, 1.0f}, {4, 0, 1.0f}, {5, 1, 1.0f}}}});
    TestGraph::MemoryGraph m2;
    m2.m_nodes.push_back(TestGraph::Node{
        .m_id = 1,
        .m_type = 1,
        .m_neighbors{std::vector<TestGraph::NeighborRecord>{{6, 1, 1.0f}, {7, 1, 1.0f}, {8, 1, 1.0f}}}});
    auto path = std::filesystem::temp_directory_path();
    TestGraph::convert(path, "0_0", std::move(m1), 2);
    TestGraph::convert(path, "1_0", std::move(m2), 2);
    snark::Graph g(path.string(), {0, 1}, snark::PartitionStorageType::memory, "");
    std::vector<snark::NodeId> nodes = {1};
    std::vector<snark::Type> types = {1};
    std::vector<size_t> sample_counts(10);
    int count = 3;
    std::vector<snark::NodeId> neighbor_nodes(count * nodes.size(), -1);
    std::vector<snark::Type> neighbor_types(count * nodes.size(), -1);
    snark::Xoroshiro128PlusGenerator gen(42);
    const size_t repetitions = 40000;
    boost::random::uniform_int_distribution<int64_t> seeds;
    std::vector<uint64_t> total_neighbor_counts(nodes.size());
    for (size_t i = 0; i < repetitions; ++i)
    {
        std::fill(std::begin(total_neighbor_counts), std::end(total_neighbor_counts), 0);
        g.UniformSampleNeighbor(false, seeds(gen), std::span(nodes), std::span(types), count, std::span(neighbor_nodes),
                                std::span(neighbor_types), std::span(total_neighbor_counts), 0, 0);
        for (auto n : neighbor_nodes)
        {
            assert(n < 10 && n >= 0);
            ++sample_counts[n];
        }
    }

    EXPECT_EQ(std::vector<size_t>({0, 0, 0, 0, 0, 30063, 29904, 30085, 29948, 0}), sample_counts);
    std::fill(std::begin(sample_counts), std::end(sample_counts), 0);
    for (size_t i = 0; i < repetitions; ++i)
    {
        std::fill(std::begin(total_neighbor_counts), std::end(total_neighbor_counts), 0);
        g.UniformSampleNeighbor(true, seeds(gen), std::span(nodes), std::span(types), count, std::span(neighbor_nodes),
                                std::span(neighbor_types), std::span(total_neighbor_counts), 0, 0);
        for (auto n : neighbor_nodes)
        {
            ++sample_counts[n];
        }
    }

    EXPECT_EQ(std::vector<size_t>({0, 0, 0, 0, 0, 30011, 30107, 29858, 30024, 0}), sample_counts);
}

TEST(GraphTest, StatisticalUniformNeighborSampleMultipleTypesNeighborsSpreadAcrossPartitions)
{
    TestGraph::MemoryGraph m1;
    m1.m_nodes.push_back(
        TestGraph::Node{.m_id = 0,
                        .m_type = 0,
                        .m_weight = 1.0f,
                        .m_neighbors{std::vector<TestGraph::NeighborRecord>{{1, 0, 1.0f}, {2, 0, 1.0f}}}});
    m1.m_nodes.push_back(TestGraph::Node{
        .m_id = 1,
        .m_type = 1,
        .m_weight = 1.0f,
        .m_neighbors{std::vector<TestGraph::NeighborRecord>{{3, 0, 1.0f}, {4, 0, 1.0f}, {5, 1, 1.0f}, {6, 1, 1.0f}}}});
    TestGraph::MemoryGraph m2;
    m2.m_nodes.push_back(TestGraph::Node{
        .m_id = 1,
        .m_type = 1,
        .m_neighbors{std::vector<TestGraph::NeighborRecord>{{7, 1, 1.0f}, {8, 1, 1.0f}, {9, 1, 1.0f}}}});
    auto path = std::filesystem::temp_directory_path();
    TestGraph::convert(path, "0_0", std::move(m1), 2);
    TestGraph::convert(path, "1_0", std::move(m2), 2);
    snark::Graph g(path.string(), {0, 1}, snark::PartitionStorageType::memory, "");
    std::vector<snark::NodeId> nodes = {1};
    std::vector<snark::Type> types = {0, 1};
    std::vector<size_t> sample_counts(10);
    int count = 2;
    std::vector<snark::NodeId> neighbor_nodes(count * nodes.size(), -1);
    std::vector<snark::Type> neighbor_types(count * nodes.size(), -1);
    snark::Xoroshiro128PlusGenerator gen(23);
    const size_t repetitions = 40000;
    boost::random::uniform_int_distribution<int64_t> seeds;
    std::vector<uint64_t> total_neighbor_counts(nodes.size());
    for (size_t i = 0; i < repetitions; ++i)
    {
        std::fill(std::begin(total_neighbor_counts), std::end(total_neighbor_counts), 0);
        g.UniformSampleNeighbor(false, seeds(gen), std::span(nodes), std::span(types), count, std::span(neighbor_nodes),
                                std::span(neighbor_types), std::span(total_neighbor_counts), 0, 0);
        for (auto n : neighbor_nodes)
        {
            ++sample_counts[n];
        }
    }

    EXPECT_EQ(std::vector<size_t>({0, 0, 0, 11684, 11256, 11449, 11355, 11404, 11328, 11524}), sample_counts);
    std::fill(std::begin(sample_counts), std::end(sample_counts), 0);
    for (size_t i = 0; i < repetitions; ++i)
    {
        std::fill(std::begin(total_neighbor_counts), std::end(total_neighbor_counts), 0);
        g.UniformSampleNeighbor(true, seeds(gen), std::span(nodes), std::span(types), count, std::span(neighbor_nodes),
                                std::span(neighbor_types), std::span(total_neighbor_counts), 0, 0);
        for (auto n : neighbor_nodes)
        {
            ++sample_counts[n];
        }
    }

    EXPECT_EQ(std::vector<size_t>({0, 0, 0, 11311, 11501, 11408, 11423, 11477, 11358, 11522}), sample_counts);
}

TEST(GraphTest, GetNeighborsMultipleTypesNeighborsSpreadAcrossPartitions)
{
    TestGraph::MemoryGraph m1;
    m1.m_nodes.push_back(
        TestGraph::Node{.m_id = 0,
                        .m_type = 0,
                        .m_weight = 1.0f,
                        .m_neighbors{std::vector<TestGraph::NeighborRecord>{{1, 0, 1.0f}, {2, 0, 1.0f}}}});
    m1.m_nodes.push_back(TestGraph::Node{
        .m_id = 1,
        .m_type = 1,
        .m_weight = 1.0f,
        .m_neighbors{std::vector<TestGraph::NeighborRecord>{{3, 0, 1.0f}, {4, 0, 1.0f}, {5, 1, 1.0f}}}});
    TestGraph::MemoryGraph m2;
    m2.m_nodes.push_back(TestGraph::Node{
        .m_id = 1, .m_type = 1, .m_neighbors{std::vector<TestGraph::NeighborRecord>{{6, 1, 1.5f}, {7, 1, 3.0f}}}});
    auto path = std::filesystem::temp_directory_path();
    TestGraph::convert(path, "0_0", std::move(m1), 2);
    TestGraph::convert(path, "1_0", std::move(m2), 2);
    snark::Graph g(path.string(), {0, 1}, snark::PartitionStorageType::memory, "");
    std::vector<snark::NodeId> nodes = {0, 1, 2};
    std::vector<snark::Type> types = {0, 1};
    std::vector<snark::NodeId> neighbor_nodes;
    std::vector<snark::Type> neighbor_types;
    std::vector<float> neighbor_weights;
    std::vector<uint64_t> neighbor_counts(nodes.size());

    g.FullNeighbor(std::span(nodes), std::span(types), neighbor_nodes, neighbor_types, neighbor_weights,
                   std::span(neighbor_counts));
    EXPECT_EQ(std::vector<snark::NodeId>({1, 2, 3, 4, 5, 6, 7}), neighbor_nodes);
    EXPECT_EQ(std::vector<snark::Type>({0, 0, 0, 0, 1, 1, 1}), neighbor_types);
    EXPECT_EQ(std::vector<uint64_t>({2, 5, 0}), neighbor_counts);
    EXPECT_EQ(std::vector<float>({1.f, 1.f, 1.f, 1.f, 1.f, 1.5f, 3.f}), neighbor_weights);
}

TEST(GraphTest, GetNodeTypesAcrossPartitions)
{
    TestGraph::MemoryGraph m1;
    for (int64_t id = 0; id < 10; ++id)
    {
        m1.m_nodes.push_back(TestGraph::Node{.m_id = id, .m_type = int32_t(id % 3)});
    }
    TestGraph::MemoryGraph m2;
    for (int64_t id = 10; id < 30; ++id)
    {
        m2.m_nodes.push_back(TestGraph::Node{.m_id = id, .m_type = int32_t(id % 3)});
    }
    auto path = std::filesystem::temp_directory_path();
    TestGraph::convert(path, "0_0", std::move(m1), 3);
    TestGraph::convert(path, "1_0", std::move(m2), 3);
    snark::Graph g(path.string(), {0, 1}, snark::PartitionStorageType::memory, "");
    std::vector<snark::NodeId> nodes = {0, 5, 10, 20, 23, 42};
    std::vector<snark::Type> types(6, -2);

    g.GetNodeType(std::span(nodes), std::span(types), -1);
    EXPECT_EQ(std::vector<snark::Type>({0, 2, 1, 2, 2, -1}), types);
}

// Neighbor Count Tests
TEST(GraphTest, GetNeigborCountSinglePartition)
{
    TestGraph::MemoryGraph m1;

    /*

    Insert first node and neighbors

    (0, 0, 1.0)
    |         |
    v         v
    (3,0,1.0)  (4,0,1.0)

    */

    m1.m_nodes.push_back(
        TestGraph::Node{.m_id = 0,
                        .m_type = 0,
                        .m_weight = 1.0f,
                        .m_neighbors{std::vector<TestGraph::NeighborRecord>{{1, 0, 1.0f}, {2, 0, 1.0f}}}});

    /*

    Insert second node and neighbors

    (1, 1, 1.0) -----> (5, 1, 1.0f)
    |         |
    v         v
    (3,0,1.0)  (4,0,1.0)

    */

    m1.m_nodes.push_back(TestGraph::Node{
        .m_id = 1,
        .m_type = 1,
        .m_weight = 1.0f,
        .m_neighbors{std::vector<TestGraph::NeighborRecord>{{3, 0, 1.0f}, {4, 0, 1.0f}, {5, 1, 1.0f}}}});

    // Initialize graph
    auto path = std::filesystem::temp_directory_path();
    TestGraph::convert(path, "0_0", std::move(m1), 2);
    snark::Graph g(path.string(), {0}, snark::PartitionStorageType::memory, "");

    // Check for singe edge type filter
    std::vector<snark::NodeId> nodes = {0, 1};
    std::vector<snark::Type> types = {0};
    std::vector<uint64_t> output_neighbors_count(nodes.size());

    g.NeighborCount(std::span(nodes), std::span(types), output_neighbors_count);
    EXPECT_EQ(std::vector<uint64_t>({2, 2}), output_neighbors_count);

    // Check for different singe edge type filter
    types = {1};
    std::fill_n(output_neighbors_count.begin(), 2, -1);

    g.NeighborCount(std::span(nodes), std::span(types), output_neighbors_count);
    EXPECT_EQ(std::vector<uint64_t>({0, 1}), output_neighbors_count);

    // Check for both edge types
    types = {0, 1};
    std::fill_n(output_neighbors_count.begin(), 2, -1);

    g.NeighborCount(std::span(nodes), std::span(types), output_neighbors_count);
    EXPECT_EQ(std::vector<uint64_t>({2, 3}), output_neighbors_count);

    // Check returns 0 for unsatisfying edge types
    types = {-1, 100};
    std::fill_n(output_neighbors_count.begin(), 2, -1);

    g.NeighborCount(std::span(nodes), std::span(types), output_neighbors_count);
    EXPECT_EQ(std::vector<uint64_t>({0, 0}), output_neighbors_count);

    // Invalid node ids
    nodes = {99, 100};
    types = {0, 1};
    std::fill_n(output_neighbors_count.begin(), 2, -1);

    g.NeighborCount(std::span(nodes), std::span(types), output_neighbors_count);
    EXPECT_EQ(std::vector<uint64_t>({0, 0}), output_neighbors_count);
}

TEST(GraphTest, GetNeigborCountMultiplePartitions)
{
    TestGraph::MemoryGraph m1;

    /*

    Insert first node and neighbors

    (0, 0, 1.0)
    |         |
    v         v
    (3,0,1.0)  (4,0,1.0)

    */

    m1.m_nodes.push_back(
        TestGraph::Node{.m_id = 0,
                        .m_type = 0,
                        .m_weight = 1.0f,
                        .m_neighbors{std::vector<TestGraph::NeighborRecord>{{1, 0, 1.0f}, {2, 0, 1.0f}}}});

    /*

    Insert second node and neighbors

    (1, 1, 1.0) -----> (5, 1, 1.0f)
    |         |
    v         v
    (3,0,1.0)  (4,0,1.0)

    */

    m1.m_nodes.push_back(TestGraph::Node{
        .m_id = 1,
        .m_type = 1,
        .m_weight = 1.0f,
        .m_neighbors{std::vector<TestGraph::NeighborRecord>{{3, 0, 1.0f}, {4, 0, 1.0f}, {5, 1, 1.0f}}}});

    /*

    Insert third node neighbors in new partition

    (1, 1, 1.0)
    |         |
    v         v
    (6,0,1.0)  (7,0,1.0)

    */

    TestGraph::MemoryGraph m2;
    m2.m_nodes.push_back(TestGraph::Node{
        .m_id = 1, .m_type = 1, .m_neighbors{std::vector<TestGraph::NeighborRecord>{{6, 1, 1.5f}, {7, 1, 3.0f}}}});

    // Initialize Graph
    auto path = std::filesystem::temp_directory_path();
    TestGraph::convert(path, "0_0", std::move(m1), 2);
    TestGraph::convert(path, "1_0", std::move(m2), 2);
    snark::Graph g(path.string(), {0, 1}, snark::PartitionStorageType::memory, "");

    // Check for singe edge type filter
    std::vector<snark::NodeId> nodes = {0, 1};
    std::vector<snark::Type> types = {1};
    std::vector<uint64_t> output_neighbors_count(nodes.size());

    g.NeighborCount(std::span(nodes), std::span(types), output_neighbors_count);
    EXPECT_EQ(std::vector<uint64_t>({0, 3}), output_neighbors_count);

    // Check for multiple edge types
    types = {0, 1};
    std::fill_n(output_neighbors_count.begin(), 2, -1);

    g.NeighborCount(std::span(nodes), std::span(types), output_neighbors_count);
    EXPECT_EQ(std::vector<uint64_t>({2, 5}), output_neighbors_count);

    // Check non-existent edge types functionality
    types = {-1, 100};
    std::fill_n(output_neighbors_count.begin(), 2, -1);

    g.NeighborCount(std::span(nodes), std::span(types), output_neighbors_count);
    EXPECT_EQ(std::vector<uint64_t>({0, 0}), output_neighbors_count);

    // Check invalid node ids handling
    nodes = {99, 100};
    types = {0, 1};
    std::fill_n(output_neighbors_count.begin(), 2, -1);

    g.NeighborCount(std::span(nodes), std::span(types), output_neighbors_count);
    EXPECT_EQ(std::vector<uint64_t>({0, 0}), output_neighbors_count);
}

INSTANTIATE_TEST_SUITE_P(StorageTypeGroup, StorageTypeGraphTest,
                         testing::Values(snark::PartitionStorageType::memory, snark::PartitionStorageType::disk));
