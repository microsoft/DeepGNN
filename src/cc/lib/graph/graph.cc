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

Graph::Graph(Metadata metadata, std::vector<std::string> paths, std::vector<uint32_t> partitions,
             PartitionStorageType storage_type)
    : m_metadata(std::move(metadata))
{
    if (paths.size() != partitions.size())
    {
        RAW_LOG_FATAL("Not enough %ld paths provided. Expected %ld for each partition.", paths.size(),
                      partitions.size());
    }

    for (size_t partition_index = 0; partition_index < paths.size(); ++partition_index)
    {
        std::vector<std::string> suffixes;
        // Go through the path folder with graph binary files.
        // For data generation flexibility we are going to load all files
        // starting with the [file_type(feat/nbs)]_[partition][anything else]
        if (!is_hdfs_path(paths[partition_index]))
        {
            for (auto &p : std::filesystem::directory_iterator(paths[partition_index]))
            {
                auto full = p.path().stem().string();
                if (full.size() <= neighbors_prefix_len)
                {
                    continue;
                }

                // Use files with neighbor lists to detect eligible suffixes.
                if (full.starts_with(neighbors_prefix) &&
                    int(partitions[partition_index]) == stoi(full.substr(neighbors_prefix_len)))
                {
                    suffixes.push_back(full.substr(neighbors_prefix_len));
                }
            }
        }
        else
        {
            auto filenames = hdfs_list_directory(paths[partition_index], m_metadata.m_config_path);
            for (auto &full : filenames)
            {
                // Use files with neighbor lists to detect eligible suffixes.
                auto loc = full.find(neighbors_prefix);
                if (loc != std::string::npos &&
                    int(partitions[partition_index]) == stoi(full.substr(loc + neighbors_prefix_len)))
                {
                    std::filesystem::path full_path = full.substr(loc + neighbors_prefix_len);
                    suffixes.push_back(full_path.stem().string());
                }
            }
        }

        // Fix loading order to obtain deterministic results for sampling.
        std::sort(std::begin(suffixes), std::end(suffixes));
        for (size_t i = 0; i < suffixes.size(); ++i)
        {
            m_partitions.emplace_back(m_metadata, paths[partition_index], suffixes[i], storage_type);
            ReadNodeMap(paths[partition_index], suffixes[i], partition_index);
        }
    }
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

                *curr_type = m_partitions[m_partitions_indices[index]].GetNodeType(m_internal_indices[index]);
                if (*curr_type != snark::PLACEHOLDER_NODE_TYPE)
                {
                    break;
                }
            }
        }
        ++curr_type;
    }
}

void Graph::GetNodeFeature(std::span<const NodeId> node_ids, std::span<const Timestamp> timestamps,
                           std::span<snark::FeatureMeta> features, std::span<uint8_t> output) const
{
    assert(std::accumulate(std::begin(features), std::end(features), size_t(0),
                           [](size_t val, const auto &f) { return val + f.second; }) *
               node_ids.size() ==
           output.size());

    const size_t features_size = output.size() / node_ids.size();
    size_t feature_offset = 0;
    absl::InlinedVector<Timestamp, 1024> feature_flags(features.size());
    for (size_t node_index = 0; node_index < node_ids.size(); ++node_index)
    {
        auto internal_id = m_node_map.find(node_ids[node_index]);
        if (internal_id == std::end(m_node_map))
        {
            std::fill_n(std::begin(output) + feature_offset, features_size, 0);
        }
        else
        {
            auto output_span = output.subspan(feature_offset, features_size);

            // We use feature_flags variable to flag features found in a partition.
            // Flags are stored as bitsets to avoid allocations.
            auto index = internal_id->second;
            size_t partition_count = m_counts[index];
            std::fill(std::begin(feature_flags), std::end(feature_flags), -1);
            for (size_t partition = 0; partition < partition_count; ++partition, ++index)
            {
                m_partitions[m_partitions_indices[index]].GetNodeFeature(
                    m_internal_indices[index],
                    timestamps.empty() ? std::nullopt : std::optional<snark::Timestamp>{timestamps[node_index]},
                    features, std::span(feature_flags), output_span);
            }

            size_t out_offset = 0;
            for (size_t feature_index = 0; feature_index < features.size(); ++feature_index)
            {
                const auto feature_size = features[feature_index].second;
                if (feature_flags[feature_index] >= 0)
                {
                    out_offset += feature_size;
                    continue;
                }

                std::fill_n(std::begin(output_span) + out_offset, feature_size, 0);
                out_offset += feature_size;
            }
        }

        feature_offset += features_size;
    }
}

void Graph::GetNodeSparseFeature(std::span<const NodeId> node_ids, std::span<const Timestamp> timestamps,
                                 std::span<const snark::FeatureId> features, std::span<int64_t> out_dimensions,
                                 std::vector<std::vector<int64_t>> &out_indices,
                                 std::vector<std::vector<uint8_t>> &out_data) const
{
    assert(features.size() == out_dimensions.size());
    assert(features.size() == out_indices.size());
    assert(features.size() == out_data.size());

    // Fill out_dimensions in case nodes don't have some features.
    std::fill(std::begin(out_dimensions), std::end(out_dimensions), 0);
    const int64_t len = node_ids.size();
    std::vector<uint64_t> values_sizes(features.size());

    absl::InlinedVector<Timestamp, 1024> feature_flags(features.size());
    for (int64_t node_index = 0; node_index < len; ++node_index)
    {
        auto internal_id = m_node_map.find(node_ids[node_index]);
        if (internal_id == std::end(m_node_map))
        {
            continue;
        }

        auto index = internal_id->second;
        size_t partition_count = m_counts[index];
        std::fill(std::begin(feature_flags), std::end(feature_flags), -1);
        for (size_t partition = 0; partition < partition_count; ++partition, ++index)
        {
            m_partitions[m_partitions_indices[index]].GetNodeSparseFeature(
                m_internal_indices[index],
                timestamps.empty() ? std::nullopt : std::optional<snark::Timestamp>{timestamps[node_index]}, features,
                std::span(feature_flags), node_index, out_dimensions, out_indices, out_data, values_sizes);
        }
    }
}

void Graph::GetNodeStringFeature(std::span<const NodeId> node_ids, std::span<const Timestamp> timestamps,
                                 std::span<const snark::FeatureId> features, std::span<int64_t> out_dimensions,
                                 std::vector<uint8_t> &out_data) const
{
    const auto features_size = features.size();
    assert(out_dimensions.size() == features_size * node_ids.size());

    const int64_t len = node_ids.size();
    absl::InlinedVector<Timestamp, 1024> feature_flags(features.size());
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
        std::fill(std::begin(feature_flags), std::end(feature_flags), -1);
        for (size_t partition = 0; partition < partition_count; ++partition, ++index)
        {
            m_partitions[m_partitions_indices[index]].GetNodeStringFeature(
                m_internal_indices[index],
                timestamps.empty() ? std::nullopt : std::optional<snark::Timestamp>{timestamps[node_index]}, features,
                std::span(feature_flags), dims_span, out_data);
        }
    }
}

void Graph::GetEdgeFeature(std::span<const NodeId> input_edge_src, std::span<const NodeId> input_edge_dst,
                           std::span<const Type> input_edge_type, std::span<const Timestamp> timestamps,
                           std::span<snark::FeatureMeta> features, std::span<uint8_t> output) const
{
    assert(std::accumulate(std::begin(features), std::end(features), size_t(0),
                           [](size_t val, const auto &f) { return val + f.second; }) *
               input_edge_src.size() ==
           output.size());

    const size_t feature_size = output.size() / input_edge_src.size();
    size_t feature_offset = 0;
    size_t edge_offset = 0;
    absl::InlinedVector<Timestamp, 1024> feature_flags(features.size());
    for (auto src_node : input_edge_src)
    {
        auto internal_id = m_node_map.find(src_node);
        if (internal_id == std::end(m_node_map))
        {
            std::fill_n(std::begin(output) + feature_offset, feature_size, 0);
        }
        else
        {
            auto index = internal_id->second;
            size_t partition_count = m_counts[index];
            std::fill(std::begin(feature_flags), std::end(feature_flags), -1);
            for (size_t partition = 0; partition < partition_count; ++partition, ++index)
            {
                m_partitions[m_partitions_indices[index]].GetEdgeFeature(
                    m_internal_indices[index], input_edge_dst[edge_offset], input_edge_type[edge_offset],
                    timestamps.empty() ? std::nullopt : std::optional<snark::Timestamp>{timestamps[edge_offset]},
                    features, std::span(feature_flags), output.subspan(feature_offset, feature_size));
            }
        }

        feature_offset += feature_size;
        ++edge_offset;
    }
}

void Graph::GetEdgeSparseFeature(std::span<const NodeId> input_edge_src, std::span<const NodeId> input_edge_dst,
                                 std::span<const Type> input_edge_type, std::span<const Timestamp> timestamps,
                                 std::span<const snark::FeatureId> features, std::span<int64_t> out_dimensions,
                                 std::vector<std::vector<int64_t>> &out_indices,
                                 std::vector<std::vector<uint8_t>> &out_values) const
{
    assert(features.size() == out_dimensions.size());
    assert(features.size() == out_indices.size());
    assert(features.size() == out_values.size());

    int64_t edge_offset = 0;
    absl::InlinedVector<Timestamp, 1024> feature_flags(features.size());
    std::vector<uint64_t> values_sizes(features.size());
    for (auto src_node : input_edge_src)
    {
        auto internal_id = m_node_map.find(src_node);
        if (internal_id != std::end(m_node_map))
        {
            auto index = internal_id->second;
            size_t partition_count = m_counts[index];
            std::fill(std::begin(feature_flags), std::end(feature_flags), -1);
            for (size_t partition = 0; partition < partition_count; ++partition, ++index)
            {
                m_partitions[m_partitions_indices[index]].GetEdgeSparseFeature(
                    m_internal_indices[index], input_edge_dst[edge_offset], input_edge_type[edge_offset],
                    timestamps.empty() ? std::nullopt : std::optional<snark::Timestamp>{timestamps[edge_offset]},
                    features, std::span(feature_flags), edge_offset, out_dimensions, out_indices, out_values,
                    values_sizes);
            }
        }

        ++edge_offset;
    }
}

void Graph::GetEdgeStringFeature(std::span<const NodeId> input_edge_src, std::span<const NodeId> input_edge_dst,
                                 std::span<const Type> input_edge_type, std::span<const Timestamp> timestamps,
                                 std::span<const snark::FeatureId> features, std::span<int64_t> out_dimensions,
                                 std::vector<uint8_t> &out_values) const
{
    const auto features_size = features.size();
    assert(features_size * input_edge_src.size() == out_dimensions.size());

    int64_t edge_offset = 0;
    absl::InlinedVector<Timestamp, 1024> feature_flags(features.size());
    for (auto src_node : input_edge_src)
    {
        auto internal_id = m_node_map.find(src_node);
        if (internal_id != std::end(m_node_map))
        {
            auto index = internal_id->second;
            size_t partition_count = m_counts[index];
            std::fill(std::begin(feature_flags), std::end(feature_flags), -1);
            for (size_t partition = 0; partition < partition_count; ++partition, ++index)
            {
                m_partitions[m_partitions_indices[index]].GetEdgeStringFeature(
                    m_internal_indices[index], input_edge_dst[edge_offset], input_edge_type[edge_offset],
                    timestamps.empty() ? std::nullopt : std::optional<snark::Timestamp>{timestamps[edge_offset]},
                    features, std::span(feature_flags),
                    out_dimensions.subspan(edge_offset * features_size, features_size), out_values);
            }
        }

        ++edge_offset;
    }
}

void Graph::NeighborCount(std::span<const NodeId> input_node_ids, std::span<const Type> input_edge_types,
                          std::span<const Timestamp> timestamps, std::span<uint64_t> output_neighbors_counts) const

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
                output_neighbors_counts[idx] += m_partitions[m_partitions_indices[index]].NeighborCount(
                    m_internal_indices[index],
                    timestamps.empty() ? std::nullopt : std::optional<snark::Timestamp>{timestamps[idx]},
                    input_edge_types);
            }
        }
    }
}

void Graph::FullNeighbor(std::span<const NodeId> input_node_ids, std::span<const Type> input_edge_types,
                         std::span<const Timestamp> timestamps, std::vector<NodeId> &output_neighbor_ids,
                         std::vector<Type> &output_neighbor_types, std::vector<float> &output_neighbors_weights,
                         std::vector<Timestamp> &output_edge_created_ts,
                         std::span<uint64_t> output_neighbors_counts) const
{
    std::fill(std::begin(output_neighbors_counts), std::end(output_neighbors_counts), 0);
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
                output_neighbors_counts[node_index] += m_partitions[m_partitions_indices[index]].FullNeighbor(
                    m_internal_indices[index],
                    timestamps.empty() ? std::nullopt : std::optional<snark::Timestamp>{timestamps[node_index]},
                    input_edge_types, output_neighbor_ids, output_neighbor_types, output_neighbors_weights,
                    output_edge_created_ts);
            }
        }
    }
}

void Graph::SampleNeighbor(int64_t seed, std::span<const NodeId> input_node_ids, std::span<Type> input_edge_types,
                           std::span<const Timestamp> timestamps, size_t count, std::span<NodeId> output_neighbor_ids,
                           std::span<Type> output_neighbor_types, std::span<float> neighbors_weights,
                           std::span<float> neighbors_total_weights, std::span<Timestamp> output_edge_created_ts,
                           NodeId default_node_id, float default_weight, Type default_edge_type) const
{
    if (!check_sorted_unique_types(input_edge_types.data(), input_edge_types.size()))
    {
        std::sort(std::begin(input_edge_types), std::end(input_edge_types));
        auto last = std::unique(std::begin(input_edge_types), std::end(input_edge_types));
        input_edge_types = input_edge_types.subspan(0, last - std::begin(input_edge_types));
    }

    for (size_t node_index = 0; node_index < input_node_ids.size(); ++node_index)
    {
        auto internal_id = m_node_map.find(input_node_ids[node_index]);
        if (internal_id == std::end(m_node_map))
        {
            std::fill_n(std::begin(output_neighbor_ids) + count * node_index, count, default_node_id);
            std::fill_n(std::begin(output_neighbor_types) + count * node_index, count, default_edge_type);
            std::fill_n(std::begin(neighbors_weights) + count * node_index, count, default_weight);
            std::fill_n(std::begin(output_edge_created_ts) + count * node_index, count, PLACEHOLDER_TIMESTAMP);
        }
        else
        {
            const auto index = internal_id->second;
            size_t partition_count = m_counts[index];
            for (size_t partition = 0; partition < partition_count; ++partition)
            {
                m_partitions[m_partitions_indices[index + partition]].SampleNeighbor(
                    seed++, m_internal_indices[index + partition],
                    timestamps.empty() ? std::nullopt : std::optional<snark::Timestamp>{timestamps[node_index]},
                    input_edge_types, count, output_neighbor_ids.subspan(count * node_index, count),
                    output_neighbor_types.subspan(count * node_index, count),
                    neighbors_weights.subspan(count * node_index, count),
                    output_edge_created_ts.subspan(count * node_index, count), neighbors_total_weights[node_index],
                    default_node_id, default_weight, default_edge_type);
            }
        }
    }
}

void Graph::UniformSampleNeighbor(bool without_replacement, int64_t seed, std::span<const NodeId> input_node_ids,
                                  std::span<Type> input_edge_types, std::span<const Timestamp> timestamps, size_t count,
                                  std::span<NodeId> output_neighbor_ids, std::span<Type> output_neighbor_types,
                                  std::span<uint64_t> neighbors_total_count,
                                  std::span<Timestamp> output_edge_created_ts, NodeId default_node_id,
                                  Type default_edge_type) const
{
    snark::Xoroshiro128PlusGenerator gen(seed);
    if (!check_sorted_unique_types(input_edge_types.data(), input_edge_types.size()))
    {
        std::sort(std::begin(input_edge_types), std::end(input_edge_types));
        auto last = std::unique(std::begin(input_edge_types), std::end(input_edge_types));
        input_edge_types = input_edge_types.subspan(0, last - std::begin(input_edge_types));
    }

    for (size_t node_index = 0; node_index < input_node_ids.size(); ++node_index)
    {
        auto internal_id = m_node_map.find(input_node_ids[node_index]);
        auto out_nb_ids = output_neighbor_ids.subspan(count * node_index, count);
        auto out_nb_types = output_neighbor_types.subspan(count * node_index, count);
        auto out_nb_ts = output_edge_created_ts.subspan(count * node_index, count);
        if (internal_id == std::end(m_node_map))
        {
            std::fill_n(std::begin(out_nb_ids), count, default_node_id);
            std::fill_n(std::begin(out_nb_types), count, default_edge_type);
            std::fill_n(std::begin(out_nb_ts), count, PLACEHOLDER_TIMESTAMP);
        }
        else
        {
            const auto index = internal_id->second;
            AlgorithmL sampler(count, gen);
            WithReplacement replacement_sampler(count, gen);
            for (size_t partition = 0; partition < m_counts[index]; ++partition)
            {
                m_partitions[m_partitions_indices[index + partition]].UniformSampleNeighbor(
                    without_replacement, m_internal_indices[index + partition],
                    timestamps.empty() ? std::nullopt : std::optional<snark::Timestamp>{timestamps[node_index]},
                    input_edge_types, count, out_nb_ids, out_nb_types, out_nb_ts, neighbors_total_count[node_index],
                    default_node_id, default_edge_type, sampler, replacement_sampler);
            }

            // We'll duplicate data if replacement is allowed and can ommit defaults.
            if (!without_replacement && neighbors_total_count[node_index] > 0)
            {
                continue;
            }

            if (neighbors_total_count[node_index] < count)
            {
                std::fill(std::begin(out_nb_ids) + neighbors_total_count[node_index], std::end(out_nb_ids),
                          default_node_id);
                std::fill(std::begin(out_nb_types) + neighbors_total_count[node_index], std::end(out_nb_types),
                          default_edge_type);
            }

            std::fill(std::begin(out_nb_ts) + (timestamps.empty() ? 0 : neighbors_total_count[node_index]),
                      std::end(out_nb_ts), snark::PLACEHOLDER_TIMESTAMP);
        }
    }
}

void Graph::LastNCreated(std::span<const NodeId> input_node_ids, std::span<Type> input_edge_types,
                         std::span<const Timestamp> input_timestamps, size_t count,
                         std::span<NodeId> output_neighbor_ids, std::span<Type> output_neighbor_types,
                         std::span<float> output_weights, std::span<Timestamp> output_timestamps,
                         NodeId default_node_id, float default_weight, Type default_edge_type,
                         Timestamp default_timestamp) const
{
    if (!check_sorted_unique_types(input_edge_types.data(), input_edge_types.size()))
    {
        std::sort(std::begin(input_edge_types), std::end(input_edge_types));
        auto last = std::unique(std::begin(input_edge_types), std::end(input_edge_types));
        input_edge_types = input_edge_types.subspan(0, last - std::begin(input_edge_types));
    }

    // backfill global timestamps with minimal values for easier management in a priority queue later.
    std::fill(std::begin(output_timestamps), std::end(output_timestamps), -1);
    for (size_t node_index = 0; node_index < input_node_ids.size(); ++node_index)
    {
        auto internal_id = m_node_map.find(input_node_ids[node_index]);
        if (internal_id == std::end(m_node_map))
        {
            std::fill_n(std::begin(output_neighbor_ids) + count * node_index, count, default_node_id);
            std::fill_n(std::begin(output_neighbor_types) + count * node_index, count, default_edge_type);
            std::fill_n(std::begin(output_weights) + count * node_index, count, default_weight);
        }
        else
        {
            const auto index = internal_id->second;
            size_t found_neighbors = 0;
            for (size_t partition = 0; partition < m_counts[index]; ++partition)
            {
                const auto partition_nbs = m_partitions[m_partitions_indices[index + partition]].LastNCreatedNeighbors(
                    m_internal_indices[index + partition], input_timestamps[node_index], input_edge_types, count,
                    output_neighbor_ids.subspan(count * node_index, count),
                    output_neighbor_types.subspan(count * node_index, count),
                    output_weights.subspan(count * node_index, count),
                    output_timestamps.subspan(count * node_index, count), default_node_id, default_edge_type,
                    default_weight, default_timestamp);
                found_neighbors = std::max(found_neighbors, partition_nbs);
            }
            if (found_neighbors < count)
            {
                const auto offset = count * node_index + found_neighbors;
                const auto backfill_count = count - found_neighbors;
                std::fill_n(std::begin(output_neighbor_ids) + offset, backfill_count, default_node_id);
                std::fill_n(std::begin(output_neighbor_types) + offset, backfill_count, default_edge_type);
                std::fill_n(std::begin(output_weights) + offset, backfill_count, default_weight);
                std::fill_n(std::begin(output_timestamps) + offset, backfill_count, default_timestamp);
            }
        }
    }
}

Metadata Graph::GetMetadata() const
{
    return m_metadata;
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
            const auto old_internal_size = m_internal_indices.size();
            m_internal_indices.resize(old_internal_size + old_count);
            std::copy_n(std::begin(m_internal_indices) + old_offset, old_count,
                        std::begin(m_internal_indices) + old_internal_size);
            m_internal_indices.emplace_back(pair[1]);
            const auto old_size = m_partitions_indices.size();
            m_partitions_indices.resize(old_size + old_count);

            std::copy_n(std::begin(m_partitions_indices) + old_offset, old_count,
                        std::begin(m_partitions_indices) + old_size);
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
