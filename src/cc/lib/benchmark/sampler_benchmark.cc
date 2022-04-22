// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <algorithm>
#include <numeric>
#include <random>
#include <string>

#include "src/cc/lib/graph/sampler.h"
#include "src/cc/lib/graph/xoroshiro.h"

#include "boost/random/uniform_real_distribution.hpp"
#include <benchmark/benchmark.h>

#ifdef SNARK_PLATFORM_LINUX
#include <mimalloc-override.h>
#endif

snark::WeightedNodeSampler create_node_sampler(size_t num_types)
{
    snark::NodeId curr_node = 0;
    snark::Xoroshiro128PlusGenerator gen(42);
    boost::random::uniform_real_distribution<float> alias_prob(0, 1);
    std::vector<std::vector<snark::WeightedNodeSamplerRecord>> records(num_types);
    const size_t nodes_per_type = 1000000;
    for (auto &t : records)
    {
        t.reserve(nodes_per_type);
        for (size_t i = 0; i < nodes_per_type; ++i)
        {
            t.emplace_back(snark::WeightedNodeSamplerRecord{curr_node, ++curr_node, alias_prob(gen)});
        }
    }
    std::vector<snark::Type> types(num_types);
    std::iota(std::begin(types), std::end(types), 0);
    std::vector<std::shared_ptr<std::vector<snark::WeightedNodeSamplerPartition>>> partitions;
    for (auto t : records)
    {
        partitions.emplace_back();
        partitions.back() = std::make_shared<std::vector<snark::WeightedNodeSamplerPartition>>();
        partitions.back()->emplace_back(std::move(t), 1.0f);
    }

    return snark::WeightedNodeSampler(types, partitions);
}

static void BM_ONE_NODE_TYPE_WEIGHTED(benchmark::State &state)
{
    auto s = create_node_sampler(1);
    size_t max_count = 1 << 13;
    std::vector<snark::NodeId> node_holder(max_count, -1);
    std::vector<snark::Type> type_holder(max_count, -1);
    std::span output_nodes(node_holder);
    std::span output_types(type_holder);
    int64_t seed = 42;
    for (auto _ : state)
    {
        const size_t batch_size = state.range(0);
        s.Sample(++seed, output_types.subspan(0, batch_size), output_nodes.subspan(0, batch_size));
    }
}

static void BM_TWO_NODE_TYPES_WEIGHTED(benchmark::State &state)
{
    auto s = create_node_sampler(2);
    size_t max_count = 1 << 13;
    std::vector<snark::NodeId> node_holder(max_count, -1);
    std::vector<snark::Type> type_holder(max_count, -1);
    std::span output_nodes(node_holder);
    std::span output_types(type_holder);
    int64_t seed = 42;
    for (auto _ : state)
    {
        const size_t batch_size = state.range(0);
        s.Sample(++seed, output_types.subspan(0, batch_size), output_nodes.subspan(0, batch_size));
    }
}

template <bool with_replacement> void RandomNodeSampling(benchmark::State &state)
{
    snark::NodeId curr_node = 0;
    const size_t num_types = 2;
    std::vector<std::vector<snark::NodeId>> records(num_types);
    const size_t nodes_per_type = 1000000;
    for (auto &t : records)
    {
        t.reserve(nodes_per_type);
        for (size_t i = 0; i < nodes_per_type; ++i)
        {
            t.emplace_back(++curr_node);
        }
    }
    std::vector<snark::Type> types;
    std::generate_n(std::back_inserter(types), num_types, [&types]() { return types.size(); });

    std::vector<std::shared_ptr<std::vector<snark::UniformNodeSamplerPartition<with_replacement>>>> partitions;
    for (auto t : records)
    {
        partitions.emplace_back();
        partitions.back() = std::make_shared<std::vector<snark::UniformNodeSamplerPartition<with_replacement>>>();
        partitions.back()->emplace_back(std::move(t));
    }

    snark::SamplerImpl<snark::UniformNodeSamplerPartition<with_replacement>> s(types, partitions);
    size_t max_count = 1 << 13;
    std::vector<snark::NodeId> node_holder(max_count, -1);
    std::vector<snark::Type> type_holder(max_count, -1);
    std::span output_nodes(node_holder);
    std::span output_types(type_holder);
    int64_t seed = 42;
    for (auto _ : state)
    {
        const size_t batch_size = state.range(0);
        s.Sample(++seed, output_types.subspan(0, batch_size), output_nodes.subspan(0, batch_size));
    }
}

static void BM_TWO_NODE_TYPES_RANDOM_WITH_REPLACEMENT(benchmark::State &state)
{
    RandomNodeSampling<true>(state);
}

static void BM_TWO_NODE_TYPES_RANDOM_WITHOUT_REPLACEMENT(benchmark::State &state)
{
    RandomNodeSampling<false>(state);
}

BENCHMARK(BM_ONE_NODE_TYPE_WEIGHTED)->RangeMultiplier(2)->Range(1 << 3, 1 << 12);
BENCHMARK(BM_TWO_NODE_TYPES_WEIGHTED)->RangeMultiplier(2)->Range(1 << 3, 1 << 12);

BENCHMARK(BM_TWO_NODE_TYPES_RANDOM_WITH_REPLACEMENT)->RangeMultiplier(2)->Range(1 << 3, 1 << 12);
BENCHMARK(BM_TWO_NODE_TYPES_RANDOM_WITHOUT_REPLACEMENT)->RangeMultiplier(2)->Range(1 << 3, 1 << 12);

BENCHMARK_MAIN();
