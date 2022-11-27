// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "graph.h"

#include <cassert>
#include <filesystem>
#include <numeric>
#include <random>
#include <string>

#include "absl/container/flat_hash_set.h"
#include <glog/logging.h>
#include <glog/raw_logging.h>

#include "locator.h"
#include "src/cc/lib/utils/thread_pool.h"
#include "types.h"

namespace snark
{
namespace
{
static const std::string neighbors_prefix = "neighbors_";
static const size_t neighbors_prefix_len = neighbors_prefix.size();

bool check_sorted_unique_types(const Type *in_edge_types, size_t count)
{
    for (size_t i = 1; i < count; ++i)
    {
        if (in_edge_types[i - 1] >= in_edge_types[i])
        {
            return false;
        }
    }

    return true;
}

} // namespace

Graph::Graph(std::string path, std::vector<uint32_t> partitions, PartitionStorageType storage_type,
             std::string config_path)
    : m_metadata(path, config_path)
{
    std::vector<std::string> suffixes;
    absl::flat_hash_set<uint32_t> partition_set(std::begin(partitions), std::end(partitions));
    // Go through the path folder with graph binary files.
    // For data generation flexibility we are going to load all files
    // starting with the [file_type(feat/nbs)]_[partition][anything else]
    if (!is_hdfs_path(path))
    {
        for (auto &p : std::filesystem::directory_iterator(path))
        {
            auto full = p.path().stem().string();

            // Use files with neighbor lists to detect eligible suffixes.
            if (full.starts_with(neighbors_prefix) &&
                partition_set.contains(std::stoi(full.substr(neighbors_prefix_len))))
            {
                suffixes.push_back(full.substr(neighbors_prefix_len));
            }
        }
    }
    else
    {
        auto filenames = hdfs_list_directory(path, m_metadata.m_config_path);
        for (auto &full : filenames)
        {
            // Use files with neighbor lists to detect eligible suffixes.
            auto loc = full.find(neighbors_prefix);
            if (loc != std::string::npos && partition_set.contains(std::stoi(full.substr(loc + neighbors_prefix_len))))
            {
                std::filesystem::path full_path = full.substr(loc + neighbors_prefix_len);
                suffixes.push_back(full_path.stem().string());
            }
        }
    }

    // Fix loading order to obtain deterministic results for sampling.
    std::sort(std::begin(suffixes), std::end(suffixes));
    m_partitions.resize(suffixes.size());
    for (size_t i = 0; i < suffixes.size(); ++i)
    {
        ReadNodeMap(path, suffixes[i], i);
    }

    ThreadPool::GetInstance().RunInParallel(
        suffixes, [this, &path, &storage_type](std::size_t index, std::string content) {
            m_partitions[index] = std::make_shared<Partition>(path, content, storage_type);
        });
}

void Graph::GetNodeType(std::span<const NodeId> node_ids, std::span<Type> output, Type default_type) const
{
    assert(output.size() == node_ids.size());
    auto curr_type = std::begin(output);
    for (auto node : node_ids)
    {
        auto internal_id = m_node_map.find(node);
        if (internal_id == std::end(m_node_map))
        {
            *curr_type = default_type;
        }
        else
        {
            auto index = internal_id->second;
            size_t partition_count = m_counts[index];
            for (size_t partition = 0; partition < partition_count; ++partition, ++index)
            {
                *curr_type = m_partitions[m_partitions_indices[index]]->GetNodeType(m_internal_indices[index]);
                if (*curr_type != snark::PLACEHOLDER_NODE_TYPE)
                    break;
            }
        }
        ++curr_type;
    }
}

void Graph::GetNodeFeature(std::span<const NodeId> node_ids, std::span<snark::FeatureMeta> features,
                           std::span<uint8_t> output) const
{
    assert(std::accumulate(std::begin(features), std::end(features), size_t(0),
                           [](size_t val, const auto &f) { return val + f.second; }) *
               node_ids.size() ==
           output.size());

    const size_t feature_size = output.size() / node_ids.size();

    // callback function to calculate a portion of the nodes in the full node_ids and then
    // process these nodes.
    auto func = [this, feature_size, &node_ids, &features](
                    const std::size_t &start_node_id, const std::size_t &end_node_id, std::span<uint8_t> &sub_span) {
        size_t feature_offset = 0;
        for (auto k = start_node_id; k < end_node_id; k++)
        {
            auto node = node_ids[k];
            auto internal_id = m_node_map.find(node);
            if (internal_id == std::end(m_node_map))
            {
                std::fill_n(std::begin(sub_span) + feature_offset, feature_size, 0);
            }
            else
            {
                auto output_span = sub_span.subspan(feature_offset, feature_size);

                auto index = internal_id->second;
                size_t partition_count = m_counts[index];
                bool found = false;
                for (size_t partition = 0; partition < partition_count && !found; ++partition, ++index)
                {
                    found = m_partitions[m_partitions_indices[index]]->GetNodeFeature(m_internal_indices[index],
                                                                                      features, output_span);
                }
            }
            feature_offset += feature_size;
        }
    };

    if (!UseThreadPoolWhenGettingFeatures(node_ids.size(), feature_size))
    {
        // process the full node list if no thread pool is used.
        func(0, node_ids.size(), output);
    }
    else
    {
        // split node list into sevaral parts and calculate each part in parallel.
        auto groups = SplitIntoGroups(node_ids.size());
        ThreadPool::GetInstance().RunInParallel(groups, [&output, feature_func = std::move(func), feature_size](
                                                            std::size_t, std::tuple<std::size_t, std::size_t> range) {
            auto offset = std::get<0>(range);
            auto length = std::get<1>(range);
            auto sub_span = output.subspan(offset * feature_size, length * feature_size);
            feature_func(offset, offset + length, sub_span);
        });
    }
}

void Graph::GetNodeSparseFeature(std::span<const NodeId> node_ids, std::span<const snark::FeatureId> features,
                                 std::span<int64_t> out_dimensions, std::vector<std::vector<int64_t>> &out_indices,
                                 std::vector<std::vector<uint8_t>> &out_data) const
{
    assert(features.size() == out_dimensions.size());
    assert(features.size() == out_indices.size());
    assert(features.size() == out_data.size());

    // Fill out_dimensions in case nodes don't have some features.
    std::fill(std::begin(out_dimensions), std::end(out_dimensions), 0);
    const int64_t len = node_ids.size();
    for (int64_t node_index = 0; node_index < len; ++node_index)
    {
        auto internal_id = m_node_map.find(node_ids[node_index]);
        if (internal_id == std::end(m_node_map))
        {
            continue;
        }

        auto index = internal_id->second;
        size_t partition_count = m_counts[index];
        bool found = false;
        for (size_t partition = 0; partition < partition_count && !found; ++partition, ++index)
        {
            found = m_partitions[m_partitions_indices[index]]->GetNodeSparseFeature(
                m_internal_indices[index], features, node_index, out_dimensions, out_indices, out_data);
        }
    }
}

void Graph::GetNodeStringFeature(std::span<const NodeId> node_ids, std::span<const snark::FeatureId> features,
                                 std::span<int64_t> out_dimensions, std::vector<uint8_t> &out_data) const
{
    const auto features_size = features.size();
    assert(out_dimensions.size() == features_size * node_ids.size());

    const int64_t len = node_ids.size();
    for (int64_t node_index = 0; node_index < len; ++node_index)
    {
        auto internal_id = m_node_map.find(node_ids[node_index]);
        if (internal_id == std::end(m_node_map))
        {
            continue;
        }

        auto dims_span = out_dimensions.subspan(features_size * node_index, features_size);

        auto index = internal_id->second;
        size_t partition_count = m_counts[index];
        bool found = false;
        for (size_t partition = 0; partition < partition_count && !found; ++partition, ++index)
        {
            found = m_partitions[m_partitions_indices[index]]->GetNodeStringFeature(m_internal_indices[index], features,
                                                                                    dims_span, out_data);
        }
    }
}

void Graph::GetEdgeFeature(std::span<const NodeId> input_edge_src, std::span<const NodeId> input_edge_dst,
                           std::span<const Type> input_edge_type, std::span<snark::FeatureMeta> features,
                           std::span<uint8_t> output) const
{
    assert(std::accumulate(std::begin(features), std::end(features), size_t(0),
                           [](size_t val, const auto &f) { return val + f.second; }) *
               input_edge_src.size() ==
           output.size());

    const size_t feature_size = output.size() / input_edge_src.size();

    // callback function to calculate a part of the input_edge_src list.
    auto func = [this, &input_edge_src, feature_size, &input_edge_dst, &input_edge_type, &features](
                    const std::size_t &start_node_id, const std::size_t &end_node_id, std::span<uint8_t> sub_output) {
        size_t feature_offset = 0;
        size_t edge_offset = start_node_id;
        for (std::size_t node_index = start_node_id; node_index < end_node_id; ++node_index)
        {
            auto src_node = input_edge_src[node_index];
            auto internal_id = m_node_map.find(src_node);
            if (internal_id == std::end(m_node_map))
            {
                std::fill_n(std::begin(sub_output) + feature_offset, feature_size, 0);
            }
            else
            {
                auto index = internal_id->second;
                size_t partition_count = m_counts[index];
                for (size_t partition = 0; partition < partition_count; ++partition, ++index)
                {
                    auto found = m_partitions[m_partitions_indices[index]]->GetEdgeFeature(
                        m_internal_indices[index], input_edge_dst[edge_offset], input_edge_type[edge_offset], features,
                        sub_output.subspan(feature_offset, feature_size));
                    if (found)
                    {
                        break;
                    }
                }
            }

            feature_offset += feature_size;
            ++edge_offset;
        }
    };

    if (!UseThreadPoolWhenGettingFeatures(input_edge_src.size(), feature_size))
    {
        func(0, input_edge_src.size(), output);
    }
    else
    {
        // divide the output into sevaral parts and each part is processed in one task.
        auto groups = SplitIntoGroups(input_edge_src.size());
        ThreadPool::GetInstance().RunInParallel(groups, [&output, feature_func = std::move(func), feature_size](
                                                            std::size_t, std::tuple<std::size_t, std::size_t> range) {
            auto offset = std::get<0>(range);
            auto length = std::get<1>(range);
            auto sub_span = output.subspan(offset * feature_size, length * feature_size);
            feature_func(offset, length, sub_span);
        });
    }
}

void Graph::GetEdgeSparseFeature(std::span<const NodeId> input_edge_src, std::span<const NodeId> input_edge_dst,
                                 std::span<const Type> input_edge_type, std::span<const snark::FeatureId> features,
                                 std::span<int64_t> out_dimensions, std::vector<std::vector<int64_t>> &out_indices,
                                 std::vector<std::vector<uint8_t>> &out_values) const
{
    assert(features.size() == out_dimensions.size());
    assert(features.size() == out_indices.size());
    assert(features.size() == out_values.size());

    int64_t edge_offset = 0;
    for (auto src_node : input_edge_src)
    {
        auto internal_id = m_node_map.find(src_node);
        if (internal_id != std::end(m_node_map))
        {
            auto index = internal_id->second;
            size_t partition_count = m_counts[index];
            for (size_t partition = 0; partition < partition_count; ++partition, ++index)
            {
                auto found = m_partitions[m_partitions_indices[index]]->GetEdgeSparseFeature(
                    m_internal_indices[index], input_edge_dst[edge_offset], input_edge_type[edge_offset], features,
                    edge_offset, out_dimensions, out_indices, out_values);
                if (found)
                {
                    break;
                }
            }
        }

        ++edge_offset;
    }
}

void Graph::GetEdgeStringFeature(std::span<const NodeId> input_edge_src, std::span<const NodeId> input_edge_dst,
                                 std::span<const Type> input_edge_type, std::span<const snark::FeatureId> features,
                                 std::span<int64_t> out_dimensions, std::vector<uint8_t> &out_values) const
{
    const auto features_size = features.size();
    assert(features_size * input_edge_src.size() == out_dimensions.size());

    int64_t edge_offset = 0;
    for (auto src_node : input_edge_src)
    {
        auto internal_id = m_node_map.find(src_node);
        if (internal_id != std::end(m_node_map))
        {
            auto index = internal_id->second;
            size_t partition_count = m_counts[index];
            for (size_t partition = 0; partition < partition_count; ++partition, ++index)
            {
                auto found = m_partitions[m_partitions_indices[index]]->GetEdgeStringFeature(
                    m_internal_indices[index], input_edge_dst[edge_offset], input_edge_type[edge_offset], features,
                    out_dimensions.subspan(edge_offset * features_size, features_size), out_values);
                if (found)
                {
                    break;
                }
            }
        }

        ++edge_offset;
    }
}

void Graph::NeighborCount(std::span<const NodeId> input_node_ids, std::span<const Type> input_edge_types,
                          std::span<uint64_t> output_neighbors_counts) const

{
    size_t num_nodes = input_node_ids.size();
    std::fill_n(std::begin(output_neighbors_counts), num_nodes, 0);

    for (size_t idx = 0; idx < num_nodes; ++idx)
    {
        auto internal_id = m_node_map.find(input_node_ids[idx]);

        if (internal_id == std::end(m_node_map))
        {
            continue;
        }
        else
        {
            auto index = internal_id->second;
            size_t partition_count = m_counts[index];

            for (size_t partition = 0; partition < partition_count; ++partition, ++index)
            {
                output_neighbors_counts[idx] += m_partitions[m_partitions_indices[index]]->NeighborCount(
                    m_internal_indices[index], input_edge_types);
            }
        }
    }
}

void Graph::FullNeighbor(std::span<const NodeId> input_node_ids, std::span<const Type> input_edge_types,
                         std::vector<NodeId> &output_neighbor_ids, std::vector<Type> &output_neighbor_types,
                         std::vector<float> &output_neighbors_weights,
                         std::span<uint64_t> output_neighbors_counts) const
{
    for (size_t node_index = 0; node_index < input_node_ids.size(); ++node_index)
    {
        auto internal_id = m_node_map.find(input_node_ids[node_index]);
        if (internal_id == std::end(m_node_map))
        {
            continue;
        }
        else
        {
            auto index = internal_id->second;
            size_t partition_count = m_counts[index];
            for (size_t partition = 0; partition < partition_count; ++partition, ++index)
            {
                output_neighbors_counts[node_index] += m_partitions[m_partitions_indices[index]]->FullNeighbor(
                    m_internal_indices[index], input_edge_types, output_neighbor_ids, output_neighbor_types,
                    output_neighbors_weights);
            }
        }
    }
}

void Graph::SampleNeighbor(int64_t seed, std::span<const NodeId> input_node_ids, std::span<Type> input_edge_types,
                           size_t count, std::span<NodeId> output_neighbor_ids, std::span<Type> output_neighbor_types,
                           std::span<float> neighbors_weights, std::span<float> neighbors_total_weights,
                           NodeId default_node_id, float default_weight, Type default_edge_type) const
{
    if (!check_sorted_unique_types(input_edge_types.data(), input_edge_types.size()))
    {
        std::sort(std::begin(input_edge_types), std::end(input_edge_types));
        auto last = std::unique(std::begin(input_edge_types), std::end(input_edge_types));
        input_edge_types = input_edge_types.subspan(0, last - std::begin(input_edge_types));
    }

    auto seed_func = [this, &input_node_ids](const std::size_t &start_node_id, const std::size_t &end_node_id,
                                             std::size_t seed_index, std::vector<std::size_t> &seed_start_per_thread) {
        for (size_t node_index = start_node_id; node_index < end_node_id; ++node_index)
        {
            auto internal_id = m_node_map.find(input_node_ids[node_index]);
            if (internal_id != std::end(m_node_map))
            {
                const auto index = internal_id->second;
                size_t partition_count = m_counts[index];
                for (size_t partition = 0; partition < partition_count; ++partition)
                {
                    seed_start_per_thread[seed_index]++;
                }
            }
        }
    };

    auto neighbor_func = [this, &input_node_ids, &output_neighbor_ids, &count, &default_node_id, &default_edge_type,
                          &default_weight, &neighbors_weights, &output_neighbor_types, &seed, &input_edge_types,
                          &neighbors_total_weights](const std::size_t &start_node_id, const std::size_t &end_node_id,
                                                    std::size_t seed_index,
                                                    std::vector<std::size_t> &seed_start_per_thread) {
        std::size_t start_seed = seed;
        if (seed_index > 0)
        {
            start_seed += std::accumulate(seed_start_per_thread.begin(), seed_start_per_thread.begin() + seed_index, 0);
        }

        for (size_t node_index = start_node_id; node_index < end_node_id; ++node_index)
        {
            auto internal_id = m_node_map.find(input_node_ids[node_index]);
            if (internal_id == std::end(m_node_map))
            {
                std::fill_n(std::begin(output_neighbor_ids) + count * node_index, count, default_node_id);
                std::fill_n(std::begin(output_neighbor_types) + count * node_index, count, default_edge_type);
                std::fill_n(std::begin(neighbors_weights) + count * node_index, count, default_weight);
            }
            else
            {
                const auto index = internal_id->second;
                size_t partition_count = m_counts[index];
                for (size_t partition = 0; partition < partition_count; ++partition)
                {
                    m_partitions[m_partitions_indices[index + partition]]->SampleNeighbor(
                        start_seed++, m_internal_indices[index + partition], input_edge_types, count,
                        output_neighbor_ids.subspan(count * node_index, count),
                        output_neighbor_types.subspan(count * node_index, count),
                        neighbors_weights.subspan(count * node_index, count), neighbors_total_weights[node_index],
                        default_node_id, default_weight, default_edge_type);
                }
            }
        }
    };

    std::vector<std::size_t> seed_start_per_thread;
    if (!UseThreadPoolWhenGettingNeighbors(input_node_ids.size(), count))
    {
        neighbor_func(0, input_node_ids.size(), 0, seed_start_per_thread);
    }
    else
    {
        auto groups = SplitIntoGroups(input_node_ids.size());
        seed_start_per_thread.resize(groups.size());
        // calculate seed for each thread
        for (size_t i = 0; i < groups.size(); i++)
        {
            seed_func(std::get<0>(groups[i]), std::get<1>(groups[i]) + std::get<0>(groups[i]), i,
                      seed_start_per_thread);
        }

        // get node neighbors using the seed calucated above.
        ThreadPool::GetInstance().RunInParallel(groups,
                                                [func = std::move(neighbor_func), &seed_start_per_thread](
                                                    std::size_t index, std::tuple<std::size_t, std::size_t> range) {
                                                    auto offset = std::get<0>(range);
                                                    auto length = std::get<1>(range);
                                                    func(offset, length + offset, index, seed_start_per_thread);
                                                });
    }
}

void Graph::UniformSampleNeighbor(bool without_replacement, int64_t seed, std::span<const NodeId> input_node_ids,
                                  std::span<Type> input_edge_types, size_t count, std::span<NodeId> output_neighbor_ids,
                                  std::span<Type> output_neighbor_types, std::span<uint64_t> neighbors_total_count,
                                  NodeId default_node_id, Type default_edge_type) const
{
    if (!check_sorted_unique_types(input_edge_types.data(), input_edge_types.size()))
    {
        std::sort(std::begin(input_edge_types), std::end(input_edge_types));
        auto last = std::unique(std::begin(input_edge_types), std::end(input_edge_types));
        input_edge_types = input_edge_types.subspan(0, last - std::begin(input_edge_types));
    }

    auto seed_func = [this, &input_node_ids](const std::size_t &start_node_id, const std::size_t &end_node_id,
                                             std::size_t seed_index, std::vector<std::size_t> &seed_start_per_thread) {
        for (size_t node_index = start_node_id; node_index < end_node_id; ++node_index)
        {
            auto internal_id = m_node_map.find(input_node_ids[node_index]);
            if (internal_id != std::end(m_node_map))
            {
                const auto index = internal_id->second;
                size_t partition_count = m_counts[index];
                for (size_t partition = 0; partition < partition_count; ++partition)
                {
                    seed_start_per_thread[seed_index]++;
                }
            }
        }
    };

    auto neighbor_func = [this, &input_node_ids, &output_neighbor_ids, &count, &default_node_id, &default_edge_type,
                          &output_neighbor_types, &seed, &input_edge_types, &without_replacement,
                          &neighbors_total_count](const std::size_t &start_node_id, const std::size_t &end_node_id,
                                                  std::size_t seed_index,
                                                  std::vector<std::size_t> &seed_start_per_thread) {
        std::size_t start_seed = seed;
        if (seed_index > 0)
        {
            start_seed += std::accumulate(seed_start_per_thread.begin(), seed_start_per_thread.begin() + seed_index, 0);
        }

        for (size_t node_index = start_node_id; node_index < end_node_id; ++node_index)
        {
            auto internal_id = m_node_map.find(input_node_ids[node_index]);
            if (internal_id == std::end(m_node_map))
            {
                std::fill_n(std::begin(output_neighbor_ids) + count * node_index, count, default_node_id);
                std::fill_n(std::begin(output_neighbor_types) + count * node_index, count, default_edge_type);
            }
            else
            {
                const auto index = internal_id->second;
                for (size_t partition = 0; partition < m_counts[index]; ++partition)
                {
                    m_partitions[m_partitions_indices[index + partition]]->UniformSampleNeighbor(
                        without_replacement, start_seed++, m_internal_indices[index + partition], input_edge_types,
                        count, output_neighbor_ids.subspan(count * node_index, count),
                        output_neighbor_types.subspan(count * node_index, count), neighbors_total_count[node_index],
                        default_node_id, default_edge_type);
                }
            }
        }
    };

    std::vector<std::size_t> seed_start_per_thread;
    if (!UseThreadPoolWhenGettingNeighbors(input_node_ids.size(), count))
    {
        neighbor_func(0, input_node_ids.size(), 0, seed_start_per_thread);
    }
    else
    {
        auto groups = SplitIntoGroups(input_node_ids.size());
        seed_start_per_thread.resize(groups.size());
        // calculate seed for each thread
        for (size_t i = 0; i < groups.size(); i++)
        {
            seed_func(std::get<0>(groups[i]), std::get<1>(groups[i]) + std::get<0>(groups[i]), i,
                      seed_start_per_thread);
        }
        // get node neighbors using the seed calucated above.
        ThreadPool::GetInstance().RunInParallel(groups,
                                                [func = std::move(neighbor_func), &seed_start_per_thread](
                                                    std::size_t index, std::tuple<std::size_t, std::size_t> range) {
                                                    auto offset = std::get<0>(range);
                                                    auto length = std::get<1>(range);
                                                    func(offset, length + offset, index, seed_start_per_thread);
                                                });
    }
}

Metadata Graph::GetMetadata() const
{
    return m_metadata;
}

bool Graph::UseThreadPoolWhenGettingFeatures(std::size_t count, std::size_t feature_size_in_byte) const
{
    return (feature_size_in_byte >= 1024 && feature_size_in_byte < 2048 && count >= 2048) ||
           (feature_size_in_byte >= 2048 && feature_size_in_byte < 4096 && count >= 1024) ||
           (feature_size_in_byte >= 4096 && feature_size_in_byte < 8192 && count >= 512) ||
           (feature_size_in_byte >= 8192 && count >= 256 && count <= 1024);
}

bool Graph::UseThreadPoolWhenGettingNeighbors(std::size_t count, std::size_t neighbor_count) const
{
    auto total = count * neighbor_count;
    return (neighbor_count < 64 && total >= 16384) ||
           (neighbor_count >= 64 && neighbor_count < 1024 && total >= 8192) ||
           (neighbor_count >= 1024 && total >= 16384);
}

std::vector<std::tuple<std::size_t, std::size_t>> Graph::SplitIntoGroups(std::size_t count, std::size_t parts) const
{
    std::vector<std::tuple<std::size_t, std::size_t>> output;

    size_t length = count / parts;
    size_t remain = count % parts;

    size_t begin = 0;
    size_t end = 0;

    for (size_t i = 0; i < std::min<size_t>(parts, count); ++i)
    {
        end += (remain > 0) ? (length + !!(remain--)) : length;
        output.push_back(std::tuple<const std::size_t, const std::size_t>(begin, end - begin));
        begin = end;
    }

    return output;
}

void Graph::ReadNodeMap(std::filesystem::path path, std::string suffix, uint32_t index)
{
    std::shared_ptr<BaseStorage<uint8_t>> node_map;
    if (!is_hdfs_path(path))
    {
        node_map = std::make_shared<DiskStorage<uint8_t>>(std::move(path), std::move(suffix), open_node_map);
    }
    else
    {
        auto full_path = path / ("node_" + suffix + ".map");
        node_map = std::make_shared<HDFSStreamStorage<uint8_t>>(full_path.c_str(), m_metadata.m_config_path);
    }
    auto node_map_ptr = node_map->start();
    size_t size = node_map->size() / 20; // 20 = 8(node_id) + 8(internal_id) + 4(node_type)
    m_node_map.reserve(size);
    m_partitions_indices.reserve(size);
    m_internal_indices.reserve(size);
    m_counts.reserve(size);
    for (size_t i = 0; i < size; ++i)
    {
        uint64_t pair[2];
        if (node_map->read(pair, 8, 2, node_map_ptr) != 2)
        {
            RAW_LOG_FATAL("Failed to read pair in a node maping");
        }

        auto el = m_node_map.find(pair[0]);
        if (el == std::end(m_node_map))
        {
            m_node_map[pair[0]] = m_internal_indices.size();
            m_internal_indices.emplace_back(pair[1]);
            m_partitions_indices.emplace_back(index);
            m_counts.emplace_back(1);
        }
        else
        {
            auto old_offset = el->second;
            auto old_count = m_counts[old_offset];
            m_node_map[pair[0]] = m_internal_indices.size();

            std::copy_n(std::begin(m_internal_indices) + old_offset, old_count, std::back_inserter(m_internal_indices));
            m_internal_indices.emplace_back(pair[1]);
            std::copy_n(std::begin(m_partitions_indices) + old_offset, old_count,
                        std::back_inserter(m_partitions_indices));
            m_partitions_indices.emplace_back(index);

            std::fill_n(std::back_inserter(m_counts), old_count + 1, old_count + 1);
        }

        assert(pair[1] == i);
        Type node_type;
        if (node_map->read(&node_type, sizeof(Type), 1, node_map_ptr) != 1)
        {
            RAW_LOG_FATAL("Failed to read node type in a node maping");
        }
    }
}

} // namespace snark
