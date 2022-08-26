// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <utility>

#include "src/cc/lib/graph/graph.h"
#include "src/cc/lib/graph/types.h"
#include "src/cc/lib/graph/xoroshiro.h"

#include "boost/random/exponential_distribution.hpp"
#include <benchmark/benchmark.h>
#ifdef SNARK_PLATFORM_LINUX
#include <mimalloc-override.h>
#endif

using NeighborRecord = std::tuple<snark::NodeId, snark::Type, float>;
struct Node
{
    int64_t m_id;
    std::vector<NeighborRecord> m_neighbors;
};

struct MemoryGraph
{
    std::vector<Node> m_nodes;
};

std::pair<size_t, size_t> create_partition(MemoryGraph t, std::filesystem::path path, std::string suffix)
{
    std::vector<NeighborRecord> edge_index;
    std::vector<uint64_t> nb_index;
    size_t counter = 0;
    int32_t node_type = 0;
    {
        std::fstream node_map(path / ("node_" + suffix + ".map"), std::ios_base::binary | std::ios_base::out);
        std::fstream node_index(path / ("node_" + suffix + ".index"), std::ios_base::binary | std::ios_base::out);
        for (auto n : t.m_nodes)
        {
            node_map.write(reinterpret_cast<const char *>(&n.m_id), 8);
            node_map.write(reinterpret_cast<const char *>(&counter), 8);
            node_map.write(reinterpret_cast<const char *>(&node_type), 4);

            node_index.write(reinterpret_cast<const char *>(&n.m_id), 8);
            node_index.write(reinterpret_cast<const char *>(&n.m_id), 8);
            ++counter;
            nb_index.push_back(edge_index.size());
            for (auto &nb : n.m_neighbors)
            {
                edge_index.emplace_back(nb);
            }
        }
        node_map.close();
        node_index.close();
    }
    {
        nb_index.push_back(edge_index.size());
        std::ofstream nb_out(path / ("neighbors_" + suffix + ".index"), std::ios_base::binary | std::ios_base::out);
        for (auto i : nb_index)
        {
            nb_out.write(reinterpret_cast<const char *>(&i), 8);
        }
        nb_out.close();
    }
    {
        std::ofstream edge_index_out(path / ("edge_" + suffix + ".index"), std::ios_base::binary | std::ios_base::out);
        for (auto &e : edge_index)
        {
            auto dst = std::get<0>(e);
            edge_index_out.write(reinterpret_cast<const char *>(&dst), 8);
            uint64_t feature_offset = 0;
            edge_index_out.write(reinterpret_cast<const char *>(&feature_offset), 8);
            auto type = std::get<1>(e);
            edge_index_out.write(reinterpret_cast<const char *>(&type), 4);
            auto weight = std::get<2>(e);
            edge_index_out.write(reinterpret_cast<const char *>(&weight), 4);
        }

        uint64_t dst = std::numeric_limits<uint64_t>::max();
        edge_index_out.write(reinterpret_cast<const char *>(&dst), 8);
        uint64_t feature_offset = 0;
        edge_index_out.write(reinterpret_cast<const char *>(&feature_offset), 8);
        int32_t type = 0;
        edge_index_out.write(reinterpret_cast<const char *>(&type), 4);
        float weight = 1;
        edge_index_out.write(reinterpret_cast<const char *>(&weight), 4);
        edge_index_out.close();
    }
    return {counter, nb_index.size()};
}

snark::Graph create_graph(size_t num_types, size_t num_nodes_per_partition, size_t num_partitions)
{
    snark::Xoroshiro128PlusGenerator gen(42);
    auto path = std::filesystem::temp_directory_path();
    std::vector<uint32_t> partitions;
    std::vector<size_t> partition_num_nodes;
    size_t num_nodes = 0;
    size_t num_edges = 0;
    // https://en.wikipedia.org/wiki/Scale-free_network
    // On average it will have 1/0.1 = 10 neighbors.
    boost::random::exponential_distribution<float> d(0.1f);
    int64_t node_id = 0;
    for (size_t p = 0; p < num_partitions; ++p)
    {
        MemoryGraph mem_graph;
        for (size_t n = 0; n < num_nodes_per_partition; ++n)
        {
            size_t num_neighbors = d(gen);
            std::vector<NeighborRecord> nbs;
            nbs.reserve(num_neighbors);
            for (size_t t = 0; t < num_types; ++t)
            {
                for (size_t i = 0; i < num_neighbors; ++i)
                {
                    nbs.emplace_back(node_id + i + t * num_neighbors, snark::Type(t), 1.0f);
                }
            }

            mem_graph.m_nodes.emplace_back(node_id++, std::move(nbs));
        }

        auto partition_nodes_edges = create_partition(std::move(mem_graph), path, std::to_string(p) + "_0");
        partition_num_nodes.emplace_back(partition_nodes_edges.first);
        num_nodes += partition_nodes_edges.first;
        num_edges += partition_nodes_edges.second;

        partitions.emplace_back(p);
    }

    {
        std::ofstream meta(path / "meta.txt");
        meta << "v" << snark::MINIMUM_SUPPORTED_VERSION << "\n";
        meta << num_nodes << "\n";
        meta << num_edges << "\n";

        meta << 1 << "\n";              // node_types_count
        meta << 1 << "\n";              // edge_types_count
        meta << 0 << "\n";              // node_features_count
        meta << 0 << "\n";              // edge_features_count
        meta << num_partitions << "\n"; // partition_count
        for (size_t partition_id = 0; partition_id < num_partitions; ++partition_id)
        {
            meta << partition_id << "\n";                      // partition id
            meta << partition_num_nodes[partition_id] << "\n"; // partition node weight
            meta << 1 << "\n";                                 // partition edge weight
        }
        meta << num_nodes << "\n";
        meta << num_edges << "\n";
        meta.close();
    }
    return snark::Graph(path, partitions, snark::PartitionStorageType::memory, "");
}

static void BM_ONE_NODE_TYPE_WEIGHTED(benchmark::State &state)
{
    const size_t num_partitions = 10;
    const size_t num_nodes_per_partition = 100000;
    auto s = create_graph(1, num_nodes_per_partition, num_partitions);
    const auto total_nodes = num_nodes_per_partition * num_partitions;
    std::vector<snark::NodeId> input_nodes(total_nodes);
    std::iota(std::begin(input_nodes), std::end(input_nodes), 0);
    std::shuffle(std::begin(input_nodes), std::end(input_nodes), snark::Xoroshiro128PlusGenerator(23));
    const size_t num_neighbors_to_sample = 10;
    size_t max_count = num_neighbors_to_sample * (1 << 13);
    std::vector<snark::Type> edge_types({0});
    std::vector<float> weight_holder(max_count, -1);
    std::vector<snark::Type> type_holder(max_count, -1);
    std::vector<snark::NodeId> node_holder(max_count, -1);
    int64_t seed = 42;
    size_t offset = 0;
    for (auto _ : state)
    {
        const size_t batch_size = state.range(0);
        std::vector<float> total_neighbor_weight(batch_size);
        s.SampleNeighbor(++seed, std::span(input_nodes).subspan(offset, batch_size), std::span(edge_types),
                         num_neighbors_to_sample,
                         std::span(node_holder).subspan(0, num_neighbors_to_sample * batch_size),
                         std::span(type_holder).subspan(0, num_neighbors_to_sample * batch_size),
                         std::span(weight_holder).subspan(0, num_neighbors_to_sample * batch_size),
                         std::span(total_neighbor_weight), 0, 0, -1);
        offset += batch_size;
        if (offset + batch_size > max_count)
        {
            offset = 0;
        }
    }
}

BENCHMARK(BM_ONE_NODE_TYPE_WEIGHTED)->RangeMultiplier(2)->Range(1 << 3, 1 << 12);
BENCHMARK_MAIN();
