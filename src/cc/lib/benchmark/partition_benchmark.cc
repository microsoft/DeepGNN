// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <random>

#include <string>

#include "src/cc/lib/graph/graph.h"
#include "src/cc/lib/graph/xoroshiro.h"
#include "src/cc/tests/mocks.h"
#include <benchmark/benchmark.h>

namespace
{
// Create a global(within benchmark code) client to share among threads.
std::shared_ptr<snark::Graph> g_client;
} // namespace

std::string create_features_graph(const size_t num_nodes, const size_t fv_size)
{
    std::string path = std::filesystem::temp_directory_path() / "benchmark_features";
    std::filesystem::create_directory(path);
    std::vector<uint32_t> partitions{0};

    TestGraph::MemoryGraph m;
    for (size_t n = 0; n < num_nodes; n++)
    {
        std::vector<float> vals(fv_size);
        std::iota(std::begin(vals), std::end(vals), n);
        m.m_nodes.push_back(TestGraph::Node{
            .m_id = snark::NodeId(n), .m_type = 0, .m_weight = 1.0f, .m_float_features = {std::move(vals)}});
    }

    TestGraph::convert(path, "0_0", std::move(m), 1);

    return path;
}

static void BM_NODE_FEATURES(benchmark::State &state, snark::PartitionStorageType storage_type)
{
    const size_t num_nodes = 100000;
    const size_t fv_size = 602;
    std::vector<snark::FeatureMeta> features = {{0, fv_size * 4}};
    std::string path;
    if (state.thread_index() == 0)
    {
        path = create_features_graph(num_nodes, fv_size);
        g_client = std::make_shared<snark::Graph>(snark::Graph(path, {0}, storage_type, ""));
    }
    const auto total_nodes = num_nodes;
    std::vector<snark::NodeId> input_nodes(total_nodes);
    std::iota(std::begin(input_nodes), std::end(input_nodes), 0);
    std::shuffle(std::begin(input_nodes), std::end(input_nodes), snark::Xoroshiro128PlusGenerator(23));
    size_t max_count = (1 << 11);
    std::vector<float> feature_holder_init(fv_size * max_count, -1);
    std::span<uint8_t> feature_holder =
        std::span(reinterpret_cast<uint8_t *>(feature_holder_init.data()), sizeof(float) * feature_holder_init.size());
    size_t offset = 0;
    for (auto _ : state)
    {
        const size_t batch_size = state.range(0);
        g_client->GetNodeFeature(std::span(input_nodes).subspan(offset, batch_size), std::span(features),
                                 std::span(feature_holder).subspan(0, fv_size * sizeof(float) * batch_size));
        offset += batch_size;
        if (offset + batch_size > max_count)
        {
            offset = 0;
        }
    }
    if (state.thread_index() == 0)
    {
        std::filesystem::remove_all(path);
    }
}

static void BM_NODE_STRING_FEATURES(benchmark::State &state, snark::PartitionStorageType storage_type)
{
    const size_t num_nodes = 100000;
    const size_t fv_size = 602;
    std::vector<snark::FeatureId> features = {0};
    std::string path;
    if (state.thread_index() == 0)
    {
        path = create_features_graph(num_nodes, fv_size);
        g_client = std::make_shared<snark::Graph>(snark::Graph(path, {0}, storage_type, ""));
    }
    const auto total_nodes = num_nodes;
    std::vector<snark::NodeId> input_nodes(total_nodes);
    std::iota(std::begin(input_nodes), std::end(input_nodes), 0);
    std::shuffle(std::begin(input_nodes), std::end(input_nodes), snark::Xoroshiro128PlusGenerator(23));
    size_t max_count = (1 << 11);
    std::vector<int64_t> dimensions(max_count);
    size_t offset = 0;
    for (auto _ : state)
    {
        std::vector<uint8_t> feature_data;
        const size_t batch_size = state.range(0);
        g_client->GetNodeStringFeature(std::span(input_nodes).subspan(offset, batch_size), std::span(features),
                                       std::span(std::begin(dimensions), std::begin(dimensions) + batch_size),
                                       feature_data);
        offset += batch_size;
        if (offset + batch_size > max_count)
        {
            offset = 0;
        }
    }
    if (state.thread_index() == 0)
    {
        std::filesystem::remove_all(path);
    }
}

static void BM_NODE_FEATURES_DISK(benchmark::State &state)
{
    BM_NODE_FEATURES(state, snark::PartitionStorageType::disk);
}

static void BM_NODE_FEATURES_MEMORY(benchmark::State &state)
{
    BM_NODE_FEATURES(state, snark::PartitionStorageType::memory);
}

static void BM_NODE_STRING_FEATURES_MEMORY(benchmark::State &state)
{
    BM_NODE_STRING_FEATURES(state, snark::PartitionStorageType::memory);
}

BENCHMARK(BM_NODE_FEATURES_DISK)->RangeMultiplier(2)->Range(1 << 3, 1 << 10)->Threads(1);
BENCHMARK(BM_NODE_FEATURES_MEMORY)->RangeMultiplier(2)->Range(1 << 3, 1 << 10)->Threads(1);
BENCHMARK(BM_NODE_FEATURES_DISK)->RangeMultiplier(2)->Range(1 << 3, 1 << 10)->Threads(2);
BENCHMARK(BM_NODE_FEATURES_MEMORY)->RangeMultiplier(2)->Range(1 << 3, 1 << 10)->Threads(2);
BENCHMARK(BM_NODE_FEATURES_DISK)->RangeMultiplier(2)->Range(1 << 3, 1 << 10)->Threads(4);
BENCHMARK(BM_NODE_FEATURES_MEMORY)->RangeMultiplier(2)->Range(1 << 3, 1 << 10)->Threads(4);
BENCHMARK(BM_NODE_STRING_FEATURES_MEMORY)->RangeMultiplier(2)->Range(1 << 3, 1 << 10)->Threads(1);
BENCHMARK(BM_NODE_STRING_FEATURES_MEMORY)->RangeMultiplier(2)->Range(1 << 3, 1 << 10)->Threads(2);
BENCHMARK(BM_NODE_STRING_FEATURES_MEMORY)->RangeMultiplier(2)->Range(1 << 3, 1 << 10)->Threads(4);
BENCHMARK_MAIN();
