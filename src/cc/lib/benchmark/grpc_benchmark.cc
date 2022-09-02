// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <memory>
#include <random>
#include <string>
#include <thread>

#include "src/cc/lib/distributed/client.h"
#include "src/cc/lib/distributed/server.h"
#include "src/cc/lib/graph/xoroshiro.h"

#include "boost/random/uniform_int_distribution.hpp"
#include "src/cc/tests/mocks.h"
#include <benchmark/benchmark.h>

#ifdef SNARK_PLATFORM_LINUX
#include <mimalloc-override.h>
#endif

namespace
{
const size_t min_batch_size = 1 << 8;
const size_t max_batch_size = 1 << 12;
const size_t num_nodes = 100000;
const size_t fv_size = 64;
} // namespace

void BM_DISTRIBUTED_GRAPH_SINGLE_NODE(benchmark::State &state)
{
    TestGraph::MemoryGraph m;
    for (size_t n = 0; n < num_nodes; n++)
    {
        std::vector<float> vals(fv_size);
        std::iota(std::begin(vals), std::end(vals), n);
        m.m_nodes.push_back(TestGraph::Node{
            .m_id = snark::NodeId(n), .m_type = 0, .m_weight = 1.0f, .m_float_features = {std::move(vals)}});
    }

    auto path = std::filesystem::temp_directory_path();
    TestGraph::convert(path, "0_0", std::move(m), 1);
    snark::GRPCServer server(std::make_shared<snark::GraphEngineServiceImpl>(path.string(), std::vector<uint32_t>{0},
                                                                             snark::PartitionStorageType::memory, ""),
                             {}, "0.0.0.0:0", {}, {}, {});
    snark::GRPCClient c({server.InProcessChannel()}, 1, 1);

    std::vector<snark::NodeId> input_nodes(num_nodes);
    std::iota(std::begin(input_nodes), std::end(input_nodes), 0);
    std::random_device rd;
    snark::Xoroshiro128PlusGenerator gen(rd());
    std::shuffle(std::begin(input_nodes), std::end(input_nodes), gen);
    std::vector<uint8_t> output(4 * fv_size * num_nodes);
    boost::random::uniform_int_distribution<size_t> distrib(0, num_nodes - (max_batch_size)-1);
    std::vector<snark::FeatureMeta> feature = {{0, fv_size}};
    for (auto _ : state)
    {
        const size_t batch_size = state.range(0);
        c.GetNodeFeature(std::span(std::begin(input_nodes) + distrib(gen), batch_size), std::span(feature),
                         std::span(std::begin(output), batch_size));
    }
}

void BM_DISTRIBUTED_GRAPH_MULTIPLE_NODES(benchmark::State &state)
{
    const size_t num_servers = 2;
    std::vector<std::unique_ptr<snark::GRPCServer>> servers;
    std::vector<std::shared_ptr<grpc::Channel>> channels;
    size_t curr_node = 0;
    for (size_t p = 0; p < num_servers; ++p)
    {
        TestGraph::MemoryGraph m;
        for (size_t n = 0; n < num_nodes / num_servers; n++, curr_node++)
        {
            std::vector<float> vals(fv_size);
            std::iota(std::begin(vals), std::end(vals), curr_node);
            m.m_nodes.push_back(TestGraph::Node{.m_id = snark::NodeId(curr_node),
                                                .m_type = 0,
                                                .m_weight = 1.0f,
                                                .m_float_features = {std::move(vals)}});
        }
        auto path = std::filesystem::temp_directory_path();
        TestGraph::convert(path, "0_0", std::move(m), 1);
        servers.emplace_back(std::make_unique<snark::GRPCServer>(
            std::make_shared<snark::GraphEngineServiceImpl>(path.string(), std::vector<uint32_t>{0},
                                                            snark::PartitionStorageType::memory, ""),
            std::shared_ptr<snark::GraphSamplerServiceImpl>{}, std::string("0.0.0.0:0"), std::string{}, std::string{},
            std::string{}));
        channels.emplace_back(servers.back()->InProcessChannel());
    }

    snark::GRPCClient c(std::move(channels), 1, 1);

    std::vector<snark::NodeId> input_nodes(num_nodes);
    std::iota(std::begin(input_nodes), std::end(input_nodes), 0);
    std::random_device rd;
    snark::Xoroshiro128PlusGenerator gen(rd());
    std::shuffle(std::begin(input_nodes), std::end(input_nodes), gen);
    std::vector<uint8_t> output(4 * fv_size * num_nodes);
    boost::random::uniform_int_distribution<size_t> distrib(0, num_nodes - (max_batch_size)-1);
    std::vector<snark::FeatureMeta> feature = {{0, 4 * fv_size}};
    for (auto _ : state)
    {
        const size_t batch_size = state.range(0);
        c.GetNodeFeature(std::span(std::begin(input_nodes) + distrib(gen), batch_size), std::span(feature),
                         std::span(std::begin(output), 4 * fv_size * batch_size));
    }
}

static void BM_REGULAR_GRAPH(benchmark::State &state)
{
    TestGraph::MemoryGraph m;
    for (size_t n = 0; n < num_nodes; n++)
    {
        std::vector<float> vals(fv_size);
        std::iota(std::begin(vals), std::end(vals), n);
        m.m_nodes.push_back(TestGraph::Node{
            .m_id = snark::NodeId(n), .m_type = 0, .m_weight = 1.0f, .m_float_features = {std::move(vals)}});
    }
    auto path = std::filesystem::temp_directory_path();
    TestGraph::convert(path, "0_0", std::move(m), 1);
    snark::Graph g(path.string(), std::vector<uint32_t>{0}, snark::PartitionStorageType::memory, "");
    std::vector<snark::NodeId> input_nodes(num_nodes);
    std::iota(std::begin(input_nodes), std::end(input_nodes), 0);
    std::random_device rd;
    snark::Xoroshiro128PlusGenerator gen(rd());
    std::shuffle(std::begin(input_nodes), std::end(input_nodes), gen);
    std::vector<uint8_t> output(4 * fv_size * num_nodes);
    boost::random::uniform_int_distribution<size_t> distrib(0, num_nodes - (max_batch_size)-1);
    std::vector<snark::FeatureMeta> feature = {{0, 4 * fv_size}};
    for (auto _ : state)
    {
        const size_t batch_size = state.range(0);

        g.GetNodeFeature(std::span(std::begin(input_nodes) + distrib(gen), batch_size), std::span(feature),
                         std::span(std::begin(output), 4 * fv_size * batch_size));
    }
}

void BM_DISTRIBUTED_SAMPLER_MULTIPLE_SERVERS(benchmark::State &state)
{
    const size_t num_servers = 2;
    std::vector<std::unique_ptr<snark::GRPCServer>> servers;
    std::vector<std::shared_ptr<grpc::Channel>> channels;
    std::vector<std::filesystem::path> dir_holders;
    const auto num_nodes_in_server = num_nodes / num_servers;
    size_t curr_node = 0;

    for (size_t p = 0; p < num_servers; ++p)
    {
        auto path = std::filesystem::temp_directory_path();

        {
            std::ofstream alias(path / "node_0_0.alias", std::ofstream::out | std::ios::binary);
            for (size_t i = 0; i < num_nodes_in_server; ++i)
            {
                alias << curr_node;
                alias << curr_node++;
                alias << 0.5f;
            }
            alias.close();
        }
        {
            std::ofstream meta(path / "meta.txt");
            meta << "v" << snark::MINIMUM_SUPPORTED_VERSION << "\n";
            meta << num_nodes_in_server << "\n";
            meta << 0 << "\n";                   // num edges
            meta << 1 << "\n";                   // node_types_count
            meta << 1 << "\n";                   // edge_types_count
            meta << 0 << "\n";                   // node_features_count
            meta << 0 << "\n";                   // edge_features_count
            meta << 1 << "\n";                   // partition_count
            meta << 0 << "\n";                   // partition id
            meta << 1 << "\n";                   // partition node weight
            meta << 1 << "\n";                   // partition edge weight
            meta << num_nodes_in_server << "\n"; // num nodes for type 0
            meta << 0 << "\n";                   // num edges for type 0
            meta.close();
        }
        servers.emplace_back(std::make_unique<snark::GRPCServer>(
            std::shared_ptr<snark::GraphEngineServiceImpl>{},
            std::make_shared<snark::GraphSamplerServiceImpl>(path, std::set<size_t>{0}), std::string("0.0.0.0:0"),
            std::string{}, std::string{}, std::string{}));

        dir_holders.emplace_back(std::move(path));
        channels.emplace_back(servers.back()->InProcessChannel());
    }

    snark::GRPCClient c(std::move(channels), 1, 1);
    std::vector<snark::NodeId> output_nodes(max_batch_size);
    std::vector<snark::Type> output_types(max_batch_size);
    std::vector<snark::Type> intput_type = {0};
    int64_t curr_seed = 0;
    auto sampler_id = c.CreateSampler(
        false, snark::CreateSamplerRequest_Category::CreateSamplerRequest_Category_WEIGHTED, std::span(intput_type));

    for (auto _ : state)
    {
        const size_t batch_size = state.range(0);
        c.SampleNodes(curr_seed++, sampler_id, std::span(output_nodes.data(), batch_size),
                      std::span(output_types.data(), batch_size));
    }
}

// Use a fixed number of iterations for easier comparison.
BENCHMARK(BM_REGULAR_GRAPH)->RangeMultiplier(4)->Range(min_batch_size, max_batch_size)->Iterations(10000);
BENCHMARK(BM_DISTRIBUTED_GRAPH_SINGLE_NODE)
    ->RangeMultiplier(4)
    ->Range(min_batch_size, max_batch_size)
    ->Iterations(10000);
BENCHMARK(BM_DISTRIBUTED_GRAPH_MULTIPLE_NODES)
    ->RangeMultiplier(4)
    ->Range(min_batch_size, max_batch_size)
    ->Iterations(10000);
BENCHMARK(BM_DISTRIBUTED_SAMPLER_MULTIPLE_SERVERS)
    ->RangeMultiplier(4)
    ->Range(min_batch_size, max_batch_size)
    ->Iterations(10000);

BENCHMARK_MAIN();
