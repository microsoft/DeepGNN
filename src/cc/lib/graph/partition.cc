// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstring>
#include <limits>
#include <queue>
#include <span>
#include <string>

#include "boost/random/binomial_distribution.hpp"
#include "boost/random/uniform_real_distribution.hpp"
#include "locator.h"
#include "partition.h"
#include "sampler.h"
#include <glog/logging.h>
#include <glog/raw_logging.h>
namespace snark
{
namespace
{
struct EdgeRecord
{
    NodeId m_dst;
    uint64_t m_feature_offset;
    Type m_type;
    float m_weight;
};

// Temporal features encoded in a following way:
// first 4 bytes - number of time intervals, num_int
// num_int * 8 bytes - start time of each interval
// num_int * 8 bytes - offsets of the corresponing time intervals in feature data.
std::tuple<uint64_t, uint64_t, Timestamp> deserialize_temporal_feature(uint64_t initial_offset, uint64_t stored_size,
                                                                       std::shared_ptr<BaseStorage<uint8_t>> storage,
                                                                       std::shared_ptr<snark::FilePtr> file_ptr,
                                                                       Timestamp timestamp)
{
    uint32_t timestamps_size = 0;
    auto timestamps_size_output = std::span(reinterpret_cast<uint8_t *>(&timestamps_size), 4);
    storage->read(initial_offset, timestamps_size_output.size(), std::begin(timestamps_size_output), file_ptr);
    if (timestamps_size == 0)
    {
        return std::make_tuple(initial_offset, 0, -1);
    }

    std::vector<int64_t> timestamp_index(2 * timestamps_size + 1);
    std::span<uint8_t> timestamp_raw(reinterpret_cast<uint8_t *>(timestamp_index.data()),
                                     timestamp_index.size() * sizeof(int64_t));
    storage->read(initial_offset + 4, timestamp_raw.size(), std::begin(timestamp_raw), file_ptr);
    auto last_timestamp = std::begin(timestamp_index) + timestamps_size;
    auto time_pos = std::lower_bound(std::begin(timestamp_index), last_timestamp, timestamp);

    // Use last known element to extract features.
    if (time_pos == last_timestamp)
    {
        --time_pos;
    }

    // Corner case: feature was created after timestamp
    if (*time_pos > timestamp && time_pos == std::begin(timestamp_index))
    {
        return std::make_tuple(initial_offset, 0, -1);
    }
    if (*time_pos > timestamp)
    {
        --time_pos;
    }

    auto offset = last_timestamp + (time_pos - std::begin(timestamp_index));
    return std::make_tuple(initial_offset + *offset, *(offset + 1) - *offset, *time_pos);
}

void deserialize_string_features(uint64_t data_offset, uint64_t stored_size,
                                 std::shared_ptr<BaseStorage<uint8_t>> storage,
                                 std::shared_ptr<snark::FilePtr> file_ptr, int64_t &out_dimension,
                                 std::vector<uint8_t> &out_values)
{
    out_dimension = stored_size;
    const auto old_values_length = out_values.size();
    out_values.resize(old_values_length + stored_size);
    auto out_values_span = std::span(out_values).subspan(old_values_length);
    storage->read(data_offset, stored_size, std::begin(out_values_span), file_ptr);
}

void deserialize_sparse_features(uint64_t data_offset, uint64_t stored_size,
                                 std::shared_ptr<BaseStorage<uint8_t>> storage, snark::FeatureId feature,
                                 std::shared_ptr<snark::FilePtr> file_ptr, int64_t prefix, int64_t &out_dimension,
                                 std::vector<int64_t> &out_indices, std::vector<uint8_t> &out_values,
                                 uint64_t &values_length)
{
    if (stored_size <=
        12) // minimum is 4 bytes to record there is a single index, actual index (8 bytes) and some data(>0 bytes).
            // Something went wrong in binary converter, we'll log a warning instead of crashing.
    {
        auto feature_string = std::to_string(feature);
        RAW_LOG_WARNING("Invalid feature request: sparse feature size is less than 12 bytes for feature %s",
                        feature_string.c_str());
        return;
    }

    assert(stored_size > 12); // minimum is 4 bytes to record there is a single index, actual index (8 bytes)
                              // and some data(>0 bytes).
    uint32_t indices_size = 0;
    auto indices_size_output = std::span(reinterpret_cast<uint8_t *>(&indices_size), 4);
    storage->read(data_offset, indices_size_output.size(), std::begin(indices_size_output), file_ptr);

    uint32_t indices_dim = 0;
    auto indices_dim_output = std::span(reinterpret_cast<uint8_t *>(&indices_dim), 4);
    storage->read(data_offset + 4, indices_dim_output.size(), std::begin(indices_dim_output), file_ptr);
    out_dimension = int64_t(indices_dim);

    assert(indices_size % indices_dim == 0);
    size_t num_values = indices_size / indices_dim;
    const auto old_len = out_indices.size();
    out_indices.resize(old_len + indices_size + num_values, prefix);
    auto output =
        std::span(reinterpret_cast<uint8_t *>(out_indices.data()) + old_len * 8, (indices_size + num_values) * 8);

    // Read expected feature_dim indices in bytes
    size_t indices_offset = data_offset + 8;
    auto curr = std::begin(output);
    for (size_t i = 0; i < num_values; ++i)
    {
        curr += 8;
        curr = storage->read(indices_offset, indices_dim * 8, curr, file_ptr);
        indices_offset += 8 * indices_dim;
    }

    // Read values
    values_length = stored_size - indices_size * 8 - 8;
    const auto old_values_length = out_values.size();
    out_values.resize(old_values_length + values_length);
    auto out_values_span = std::span(out_values).subspan(old_values_length);
    storage->read(indices_offset, values_length, std::begin(out_values_span), file_ptr);
}

} // namespace
Partition::Partition(Metadata metadata, std::filesystem::path path, std::string suffix,
                     PartitionStorageType storage_type)
    : m_metadata(std::move(metadata)), m_storage_type(storage_type)
{
    ReadNodeMap(path, suffix);
    ReadNodeFeatures(path, suffix);
    ReadEdges(std::move(path), std::move(suffix));
}

void Partition::ReadNodeMap(std::filesystem::path path, std::string suffix)
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
    size_t size = node_map->size() / 20;
    m_node_types.reserve(size);
    for (size_t i = 0; i < size; ++i)
    {
        uint64_t pair[2];
        if (node_map->read(pair, 8, 2, node_map_ptr) != 2)
        {
            RAW_LOG_FATAL("Failed to read pair in a node maping");
        }
        Type node_type;
        if (node_map->read(&node_type, 4, 1, node_map_ptr) != 1)
        {
            RAW_LOG_FATAL("Failed to read node type in a node maping");
        }
        m_node_types.emplace_back(node_type);
    }
}

void Partition::ReadEdges(std::filesystem::path path, std::string suffix)
{
    ReadNeighborsIndex(path, suffix);
    ReadEdgeIndex(path, suffix);
    if (m_metadata.m_watermark >= 0)
    {
        ReadEdgeTimestamps(path, suffix);
    }

    if (m_metadata.m_edge_feature_count > 0)
    {
        ReadEdgeFeaturesIndex(path, suffix);
        ReadEdgeFeaturesData(std::move(path), std::move(suffix));
    }
    else
    {
        m_edge_features = std::make_shared<MemoryStorage<uint8_t>>(path, suffix, nullptr);
    }
}

void Partition::ReadNeighborsIndex(std::filesystem::path path, std::string suffix)
{
    std::shared_ptr<BaseStorage<uint8_t>> neighbors_index;
    if (!is_hdfs_path(path))
    {
        neighbors_index =
            std::make_shared<DiskStorage<uint8_t>>(std::move(path), std::move(suffix), open_neighbor_index);
    }
    else
    {
        auto full_path = path / ("neighbors_" + suffix + ".index");
        neighbors_index = std::make_shared<HDFSStreamStorage<uint8_t>>(full_path.c_str(), m_metadata.m_config_path);
    }
    auto neighbors_index_ptr = neighbors_index->start();
    size_t size_64 = neighbors_index->size() / 8;
    m_neighbors_index.resize(size_64);
    if (size_64 != neighbors_index->read(m_neighbors_index.data(), 8, size_64, neighbors_index_ptr))
    {
        RAW_LOG_FATAL("Failed to read neighbor index file");
    }
}

void Partition::ReadEdgeTimestamps(std::filesystem::path path, std::string suffix)
{
    std::shared_ptr<BaseStorage<uint8_t>> edge_timestamps;
    if (!is_hdfs_path(path))
    {
        edge_timestamps =
            std::make_shared<DiskStorage<uint8_t>>(std::move(path), std::move(suffix), open_edge_timestamps);
    }
    else
    {
        auto full_path = path / ("edge_" + suffix + ".timestamp");
        edge_timestamps = std::make_shared<HDFSStreamStorage<uint8_t>>(full_path.c_str(), m_metadata.m_config_path);
    }

    auto edge_timestamps_ptr = edge_timestamps->start();
    size_t size_64 = edge_timestamps->size() / 8;
    if (size_64 == 0)
    {
        return;
    }

    size_64--;
    m_edge_timestamps.resize(size_64 / 2);
    assert(m_edge_timestamps.size() + 1 ==
           m_edge_destination.size()); // destination is padded with an extra value for faster search.
    if (1 != edge_timestamps->read(&m_watermark, 8, 1, edge_timestamps_ptr))
    {
        RAW_LOG_FATAL("Failed to read watermark from edge timestamps file");
    }

    for (auto &ts : m_edge_timestamps)
    {
        if (1 != edge_timestamps->read(&ts.first, 8, 1, edge_timestamps_ptr))
        {
            RAW_LOG_FATAL("Failed to read edge timestamps file");
        }

        if (1 != edge_timestamps->read(&ts.second, 8, 1, edge_timestamps_ptr))
        {
            RAW_LOG_FATAL("Failed to read edge timestamps file");
        }
    }
}

void Partition::ReadEdgeIndex(std::filesystem::path path, std::string suffix)
{
    assert(sizeof(EdgeRecord) == (sizeof(NodeId) + sizeof(uint64_t) + sizeof(Type) + sizeof(float)));
    std::shared_ptr<BaseStorage<uint8_t>> edge_index;
    if (!is_hdfs_path(path))
    {
        edge_index = std::make_shared<DiskStorage<uint8_t>>(std::move(path), std::move(suffix), open_edge_index);
    }
    else
    {
        auto full_path = path / ("edge_" + suffix + ".index");
        edge_index = std::make_shared<HDFSStreamStorage<uint8_t>>(full_path.c_str(), m_metadata.m_config_path);
    }
    auto edge_index_ptr = edge_index->start();
    size_t num_edges = edge_index->size() / sizeof(EdgeRecord);
    m_edge_weights.reserve(num_edges);
    size_t next = 1;
    for (size_t curr_src = 0; next < m_neighbors_index.size(); ++curr_src, ++next)
    {
        size_t start_offset = m_neighbors_index[curr_src];
        size_t end_offset = m_neighbors_index[next];
        m_neighbors_index[curr_src] = m_edge_types.size();
        size_t nb_count = end_offset - start_offset;
        if (nb_count == 0)
        {
            continue;
        }
        Type curr_type = -1;
        // Accumulate weights for faster binary search in sampling
        float acc_weight = 0;
        for (size_t curr = start_offset; curr < end_offset; ++curr)
        {
            EdgeRecord edge;
            if (1 != edge_index->read(&edge, sizeof(EdgeRecord), 1, edge_index_ptr))
            {
                RAW_LOG_FATAL("Failed to read edge index file");
            }
            if (edge.m_type != curr_type)
            {
                curr_type = edge.m_type;
                m_edge_types.emplace_back(curr_type);
                m_edge_type_offset.emplace_back(m_edge_destination.size());
                acc_weight = 0;
            }
            m_edge_destination.push_back(edge.m_dst);
            acc_weight += edge.m_weight;
            m_edge_weights.push_back(acc_weight);
            if (m_metadata.m_edge_feature_count > 0)
            {
                m_edge_feature_offset.push_back(edge.m_feature_offset);
            }
        }
    }
    EdgeRecord edge;
    if (1 != edge_index->read(&edge, sizeof(EdgeRecord), 1, edge_index_ptr))
    {
        RAW_LOG_FATAL("Failed to read edge index file");
    }
    // Extra padding to simplify edge type count calculations.
    m_neighbors_index.back() = m_edge_types.size();
    m_edge_types.push_back(edge.m_type);
    m_edge_type_offset.push_back(m_edge_destination.size());
    m_edge_destination.push_back(edge.m_dst);
    if (m_metadata.m_edge_feature_count > 0)
    {
        m_edge_feature_offset.push_back(edge.m_feature_offset);
    }
}

void Partition::ReadNodeFeatures(std::filesystem::path path, std::string suffix)
{
    ReadNodeIndex(path, suffix);
    if (m_metadata.m_node_feature_count == 0)
    {
        // It's ok to miss files if there are no features.
        m_node_features = std::make_shared<MemoryStorage<uint8_t>>(path, suffix, nullptr);
        return;
    }
    ReadNodeFeaturesIndex(path, suffix);
    ReadNodeFeaturesData(path, suffix);
}

void Partition::ReadNodeIndex(std::filesystem::path path, std::string suffix)
{
    std::shared_ptr<BaseStorage<uint8_t>> node_index;
    if (!is_hdfs_path(path))
    {
        node_index = std::make_shared<DiskStorage<uint8_t>>(std::move(path), std::move(suffix), open_node_index);
    }
    else
    {
        auto full_path = path / ("node_" + suffix + ".index");
        node_index = std::make_shared<HDFSStreamStorage<uint8_t>>(full_path.c_str(), m_metadata.m_config_path);
    }
    auto node_index_ptr = node_index->start();
    size_t size = node_index->size() / 8;
    m_node_index.resize(size);
    if (size != node_index->read(m_node_index.data(), 8, size, node_index_ptr))
    {
        RAW_LOG_FATAL("Failed to read node index file");
    }
}

void Partition::ReadNodeFeaturesIndex(std::filesystem::path path, std::string suffix)
{
    std::shared_ptr<BaseStorage<uint8_t>> node_features_index;
    if (!is_hdfs_path(path))
    {
        node_features_index =
            std::make_shared<DiskStorage<uint8_t>>(std::move(path), std::move(suffix), open_node_features_index);
    }
    else
    {
        auto full_path = path / ("node_features_" + suffix + ".index");
        node_features_index = std::make_shared<HDFSStreamStorage<uint8_t>>(full_path.c_str(), m_metadata.m_config_path);
    }
    auto node_features_index_ptr = node_features_index->start();
    size_t size = node_features_index->size() / 8;
    m_node_feature_index.resize(size);
    if (size != node_features_index->read(m_node_feature_index.data(), 8, size, node_features_index_ptr))
    {
        RAW_LOG_FATAL("Failed to read node feature index file");
    }
}

void Partition::ReadNodeFeaturesData(std::filesystem::path path, std::string suffix)
{
    if (is_hdfs_path(path))
    {
        auto full_path = path / ("node_features_" + suffix + ".data");
        m_node_features = std::make_shared<HDFSStorage<uint8_t>>(full_path.c_str(), m_metadata.m_config_path,
                                                                 std::move(suffix), &open_node_features_data);
    }
    else if (m_storage_type == PartitionStorageType::memory)
    {
        m_node_features =
            std::make_shared<MemoryStorage<uint8_t>>(std::move(path), std::move(suffix), &open_node_features_data);
    }
    else if (m_storage_type == PartitionStorageType::disk)
    {
        m_node_features =
            std::make_shared<DiskStorage<uint8_t>>(std::move(path), std::move(suffix), &open_node_features_data);
    }
}

void Partition::ReadEdgeFeaturesIndex(std::filesystem::path path, std::string suffix)
{
    std::shared_ptr<BaseStorage<uint8_t>> edge_features_index;
    if (!is_hdfs_path(path))
    {
        edge_features_index =
            std::make_shared<DiskStorage<uint8_t>>(std::move(path), std::move(suffix), open_edge_features_index);
    }
    else
    {
        auto full_path = path / ("edge_features_" + suffix + ".index");
        edge_features_index = std::make_shared<HDFSStreamStorage<uint8_t>>(full_path.c_str(), m_metadata.m_config_path);
    }
    auto edge_features_index_ptr = edge_features_index->start();
    size_t size = edge_features_index->size();
    size_t size_64 = size / 8;
    m_edge_feature_index.resize(size_64);
    if (size != edge_features_index->read(m_edge_feature_index.data(), 1, size, edge_features_index_ptr))
    {
        RAW_LOG_FATAL("Failed to read node feature index file");
    }
}

void Partition::ReadEdgeFeaturesData(std::filesystem::path path, std::string suffix)
{
    if (is_hdfs_path(path))
    {
        auto full_path = path / ("edge_features_" + suffix + ".data");
        m_edge_features = std::make_shared<HDFSStorage<uint8_t>>(full_path.c_str(), m_metadata.m_config_path,
                                                                 std::move(suffix), &open_edge_features_data);
    }
    else if (m_storage_type == PartitionStorageType::memory)
    {
        m_edge_features =
            std::make_shared<MemoryStorage<uint8_t>>(std::move(path), std::move(suffix), &open_edge_features_data);
    }
    else if (m_storage_type == PartitionStorageType::disk)
    {
        m_edge_features =
            std::make_shared<DiskStorage<uint8_t>>(std::move(path), std::move(suffix), &open_edge_features_data);
    }
}

Type Partition::GetNodeType(uint64_t internal_node_id) const
{
    return m_node_types[internal_node_id];
}

void Partition::GetNodeFeature(uint64_t internal_id, std::optional<Timestamp> node_ts,
                               std::span<snark::FeatureMeta> features, std::span<Timestamp> feature_flags,
                               std::span<uint8_t> output) const
{
    if (m_node_feature_index.empty() || !m_node_features)
    {
        return;
    }

    auto file_ptr = m_node_features->start();
    auto curr = std::begin(output);
    auto feature_index_offset = m_node_index[internal_id];
    auto next_offset = m_node_index[internal_id + 1];

    for (size_t feature_index = 0; feature_index < features.size(); ++feature_index)
    {
        if ((!node_ts.has_value() && feature_flags[feature_index] >= 0) ||
            (node_ts.has_value() && feature_flags[feature_index] == node_ts.value()))
        {
            curr += features[feature_index].second;
            continue;
        }

        const auto feature_id = features[feature_index].first;
        const auto feature_size = features[feature_index].second;

        // Requested feature_id is larger than known features, fill with 0s later.
        if (next_offset - feature_index_offset <= uint64_t(feature_id))
        {
            curr += features[feature_index].second;
            continue;
        }

        Timestamp feature_ts = 0;
        auto data_offset = m_node_feature_index[feature_index_offset + feature_id];
        auto stored_size = m_node_feature_index[feature_index_offset + feature_id + 1] - data_offset;
        if (node_ts.has_value())
        {
            std::tie(data_offset, stored_size, feature_ts) =
                deserialize_temporal_feature(data_offset, stored_size, m_node_features, file_ptr, node_ts.value());
        }

        // Skip old feature in a partition.
        if (feature_flags[feature_index] >= feature_ts)
        {
            curr += feature_size;
            continue;
        }

        feature_flags[feature_index] = feature_ts;
        curr = m_node_features->read(data_offset, std::min<uint64_t>(feature_size, stored_size), curr, file_ptr);
        if (stored_size < feature_size)
        {
            curr = std::fill_n(curr, feature_size - stored_size, 0);
        }
    }
}

void Partition::GetNodeSparseFeature(uint64_t internal_node_id, std::optional<Timestamp> node_ts,
                                     std::span<const snark::FeatureId> features, std::span<Timestamp> feature_flags,
                                     int64_t prefix, std::span<int64_t> out_dimensions,
                                     std::vector<std::vector<int64_t>> &out_indices,
                                     std::vector<std::vector<uint8_t>> &out_values,
                                     std::vector<uint64_t> &values_sizes) const
{
    assert(features.size() == out_dimensions.size());
    if (m_node_feature_index.empty() || !m_node_features)
    {
        return;
    }
    auto file_ptr = m_node_features->start();
    auto feature_index_offset = m_node_index[internal_node_id];
    auto next_offset = m_node_index[internal_node_id + 1];
    for (size_t feature_index = 0; feature_index < features.size(); ++feature_index)
    {
        if ((!node_ts.has_value() && feature_flags[feature_index] >= 0) ||
            (node_ts.has_value() && feature_flags[feature_index] == node_ts.value()))
        {
            continue;
        }

        const auto feature_id = features[feature_index];
        // Requested feature_id is larger than known features, fill with 0s later.
        if (next_offset - feature_index_offset <= uint64_t(feature_id))
        {
            continue;
        }

        Timestamp feature_ts = 0;
        auto data_offset = m_node_feature_index[feature_index_offset + feature_id];
        auto stored_size = m_node_feature_index[feature_index_offset + feature_id + 1] - data_offset;
        // Check if the feature is empty
        if (stored_size == 0)
        {
            continue;
        }

        if (node_ts.has_value())
        {
            std::tie(data_offset, stored_size, feature_ts) =
                deserialize_temporal_feature(data_offset, stored_size, m_node_features, file_ptr, node_ts.value());
        }

        // Skip old feature in a partition.
        if (feature_flags[feature_index] >= feature_ts)
        {
            continue;
        }

        // Overwrite old string features.
        if (feature_flags[feature_index] >= 0)
        {
            const auto new_indices_size = out_indices[feature_index].size() - out_dimensions[feature_index] - 1;
            out_indices[feature_index].resize(new_indices_size);
            const auto new_values_size = out_values[feature_index].size() - values_sizes[feature_index];
            out_values[feature_index].resize(new_values_size);
        }

        feature_flags[feature_index] = feature_ts;

        deserialize_sparse_features(data_offset, stored_size, m_node_features, feature_id, file_ptr, prefix,
                                    out_dimensions[feature_index], out_indices[feature_index],
                                    out_values[feature_index], values_sizes[feature_index]);
    }
}

void Partition::GetNodeStringFeature(uint64_t internal_node_id, std::optional<Timestamp> node_ts,
                                     std::span<const snark::FeatureId> features, std::span<Timestamp> feature_flags,
                                     std::span<int64_t> out_dimensions, std::vector<uint8_t> &out_values) const
{
    assert(features.size() == out_dimensions.size());

    if (m_node_feature_index.empty() || m_node_features == nullptr)
    {
        return;
    }
    auto file_ptr = m_node_features->start();
    auto feature_index_offset = m_node_index[internal_node_id];
    auto next_offset = m_node_index[internal_node_id + 1];
    for (size_t feature_index = 0; feature_index < features.size(); ++feature_index)
    {
        if ((!node_ts.has_value() && feature_flags[feature_index] >= 0) ||
            (node_ts.has_value() && feature_flags[feature_index] == node_ts.value()))
        {
            continue;
        }

        const auto feature_id = features[feature_index];

        // Requested feature_id is larger than known features, fill with 0s later.
        if (next_offset - feature_index_offset <= uint64_t(feature_id) || m_node_feature_index.empty())
        {
            continue;
        }

        auto data_offset = m_node_feature_index[feature_index_offset + feature_id];
        auto stored_size = m_node_feature_index[feature_index_offset + feature_id + 1] - data_offset;
        Timestamp feature_ts = 0;

        // Check if the feature is empty
        if (stored_size == 0)
        {
            continue;
        }

        if (node_ts.has_value())
        {
            std::tie(data_offset, stored_size, feature_ts) =
                deserialize_temporal_feature(data_offset, stored_size, m_node_features, file_ptr, node_ts.value());
        }

        // Skip old feature in a partition.
        if (feature_flags[feature_index] >= feature_ts)
        {
            continue;
        }

        // Overwrite old string features.
        if (feature_flags[feature_index] >= 0)
        {
            const auto new_size = out_values.size() - out_dimensions[feature_index];
            out_values.resize(new_size);
        }

        feature_flags[feature_index] = feature_ts;
        deserialize_string_features(data_offset, stored_size, m_node_features, file_ptr, out_dimensions[feature_index],
                                    out_values);
    }
}

template <class F>
size_t NeighborIndexIterator(uint64_t internal_id, std::optional<Timestamp> node_ts, std::span<const Type> edge_types,
                             F func, const std::vector<uint64_t> &m_neighbors_index,
                             const std::vector<Type> &m_edge_types, const std::vector<uint64_t> &m_edge_type_offset)

{
    const auto offset = m_neighbors_index[internal_id];
    const auto nb_count = m_neighbors_index[internal_id + 1] - offset;

    // Check if node doesn't have any neighbors
    if (nb_count == 0)
    {
        return 0;
    }

    size_t result = 0;
    size_t curr_type = 0;
    for (size_t i = offset; i < offset + nb_count; ++i)
    {
        for (; curr_type < edge_types.size() && edge_types[curr_type] < m_edge_types[i]; ++curr_type)
        {
        }
        if (curr_type == edge_types.size())
        {
            break;
        }
        for (; i < offset + nb_count && edge_types[curr_type] > m_edge_types[i]; ++i)
        {
        }
        if (i == offset + nb_count)
        {
            break;
        }
        // Find satisfying edge type, if exists
        if (m_edge_types[i] == edge_types[curr_type])
        {
            const auto start = m_edge_type_offset[i];
            const auto last = m_edge_type_offset[i + 1];

            result += last - start;
            func(start, last, i);
        }
    }
    return result;
}

namespace
{

auto find_start(const std::span<const std::pair<Timestamp, Timestamp>> &range, Timestamp ts)
{
    if (range.empty())
    {
        return std::end(range);
    }

    if (range.front().first <= ts && range.front().second > ts)
    {
        return std::begin(range);
    }

    ts = std::max(ts, range.front().second);
    return std::lower_bound(
        std::begin(range), std::end(range), ts,
        [](const std::pair<Timestamp, Timestamp> &a, int x) { return a.second != -1 && a.second <= x; });
}

using cspan_it = std::span<const std::pair<Timestamp, Timestamp>>::iterator;

cspan_it find_last(const std::span<const std::pair<Timestamp, Timestamp>> &range, int ts)
{
    if (range.front().first > ts)
    {
        return std::begin(range);
    }

    int last = range.front().second;
    return std::upper_bound(
        std::begin(range), std::end(range), ts,
        [last](int ts, const std::pair<Timestamp, Timestamp> &a) { return a.second > last || a.first > ts; });
}

} // anonymous namespace

size_t Partition::NeighborCount(uint64_t internal_id, std::optional<Timestamp> node_ts,
                                std::span<const Type> edge_types) const
{
    if (!node_ts)
    {
        auto lambda = [](const auto start, const auto last, const auto i) {};
        return NeighborIndexIterator(internal_id, std::move(node_ts), edge_types, std::move(lambda), m_neighbors_index,
                                     m_edge_types, m_edge_type_offset);
    }

    size_t count = 0;
    auto lambda = [&count, node_ts, this](const auto start, const auto last, const auto i) {
        auto timestamp_span = std::span(m_edge_timestamps).subspan(start, last - start);
        auto start_dist = start;
        for (auto it = find_start(timestamp_span, node_ts.value());
             it != std::end(timestamp_span) && !(it->second == -1 && it->first > node_ts.value());
             it = find_start(timestamp_span, node_ts.value()))
        {
            auto diff = it - std::begin(timestamp_span);
            auto local_ts = timestamp_span.subspan(diff);
            auto local_last = find_last(local_ts, node_ts.value());

            const auto local_diff = size_t(local_last - std::begin(local_ts));
            count += local_diff;
            if (local_diff == 0)
            {
                ++local_last;
            }
            start_dist += diff + size_t(local_last - std::begin(local_ts));
            timestamp_span = timestamp_span.subspan(local_last - std::begin(local_ts) + diff);
        }
    };

    NeighborIndexIterator(internal_id, std::move(node_ts), edge_types, std::move(lambda), m_neighbors_index,
                          m_edge_types, m_edge_type_offset);
    return count;
}

size_t Partition::FullNeighbor(uint64_t internal_id, std::optional<Timestamp> node_ts, std::span<const Type> edge_types,
                               std::vector<NodeId> &out_neighbors_ids, std::vector<Type> &out_edge_types,
                               std::vector<float> &out_edge_weights, std::vector<Timestamp> &out_edge_created_ts) const
{
    if (node_ts)
    {
        size_t count = 0;
        auto lambda = [&count, node_ts = node_ts.value(), &ts = m_edge_timestamps, &out_neighbors_ids, &out_edge_types,
                       &out_edge_weights, &out_edge_created_ts, this](const auto start, const auto last, const auto i) {
            for (size_t curr_ts = start; curr_ts < last; ++curr_ts)
            {
                if (ts[curr_ts].first <= node_ts && (ts[curr_ts].second > node_ts || ts[curr_ts].second == -1))
                {
                    out_neighbors_ids.emplace_back(m_edge_destination[curr_ts]);
                    out_edge_weights.emplace_back(curr_ts > start
                                                      ? m_edge_weights[curr_ts] - m_edge_weights[curr_ts - 1]
                                                      : m_edge_weights[start]);
                    out_edge_types.emplace_back(m_edge_types[i]);
                    out_edge_created_ts.emplace_back(ts[curr_ts].first);
                    ++count;
                }
            }
        };

        NeighborIndexIterator(internal_id, std::move(node_ts), edge_types, std::move(lambda), m_neighbors_index,
                              m_edge_types, m_edge_type_offset);
        return count;
    }

    auto lambda = [&out_neighbors_ids, &out_edge_types, &out_edge_weights, &out_edge_created_ts,
                   this](auto start, auto last, int i) {
        // m_edge_destination[last-1]+1 - take the last element and then advance the pointer
        // to imitate std::end, otherwise we'll have an out of range exception.
        out_neighbors_ids.insert(std::end(out_neighbors_ids), &m_edge_destination[start],
                                 &m_edge_destination[last - 1] + 1);
        auto original_type_size = out_edge_types.size();
        out_edge_types.resize(original_type_size + last - start, m_edge_types[i]);
        out_edge_created_ts.resize(original_type_size + last - start, snark::PLACEHOLDER_TIMESTAMP);
        out_edge_weights.reserve(out_edge_weights.size() + last - start);
        for (size_t index = start; index < last; ++index)
        {
            out_edge_weights.emplace_back(index > start ? m_edge_weights[index] - m_edge_weights[index - 1]
                                                        : m_edge_weights[start]);
        }
    };

    return NeighborIndexIterator(internal_id, std::move(node_ts), edge_types, std::move(lambda), m_neighbors_index,
                                 m_edge_types, m_edge_type_offset);
}

size_t Partition::LastNCreatedNeighbors(uint64_t internal_node_id, Timestamp timestamp,
                                        std::span<const Type> in_edge_types, uint64_t count,
                                        std::span<NodeId> out_nodes, std::span<Type> out_types,
                                        std::span<float> out_edge_weights, std::span<Timestamp> out_timestamps,
                                        NodeId default_node_id, Type default_edge_type, float default_weight,
                                        Timestamp default_timestamp) const
{
    using ts_position = std::pair<Timestamp, size_t>;
    std::priority_queue<ts_position, std::vector<ts_position>, std::greater<ts_position>> lastn;
    for (size_t i = 0; i < size_t(count); ++i)
    {
        const auto ts = out_timestamps[i];
        if (ts < 0)
        {
            break;
        }
        lastn.emplace(ts, i);
    }
    auto lambda = [&out_nodes, &out_types, &out_edge_weights, &out_timestamps, &lastn, timestamp, count,
                   this](auto start, auto last, int i) {
        for (size_t index = start; index < last; ++index)
        {
            auto &ts = m_edge_timestamps[index];
            if ((ts.second != -1 && ts.second < timestamp) || ts.first > timestamp)
            {
                continue;
            }

            size_t pos = lastn.size();
            if (lastn.size() == count)
            {
                auto top = lastn.top();
                if (top.first >= ts.first)
                {
                    continue;
                }
                else
                {
                    lastn.pop();
                }

                pos = top.second;
            }

            lastn.emplace(ts.first, pos);
            out_nodes[pos] = m_edge_destination[index];
            out_types[pos] = m_edge_types[i];
            out_timestamps[pos] = ts.first;
            out_edge_weights[pos] = m_edge_weights[index];
            if (index > start)
            {
                out_edge_weights[pos] -= m_edge_weights[index - 1];
            }
        }
    };

    NeighborIndexIterator(internal_node_id, timestamp, in_edge_types, std::move(lambda), m_neighbors_index,
                          m_edge_types, m_edge_type_offset);
    return lastn.size();
}

std::optional<size_t> Partition::EdgeFeatureOffset(uint64_t internal_src_node_id, NodeId input_edge_dst,
                                                   Type input_edge_type) const
{
    auto file_ptr = m_edge_features->start();
    const auto offset = m_neighbors_index[internal_src_node_id];
    const auto nb_count = m_neighbors_index[internal_src_node_id + 1] - offset;

    // Check if node doesn't have any neighbors
    if (nb_count == 0)
    {
        return std::nullopt;
    }

    auto type_offset = std::numeric_limits<size_t>::max();
    for (size_t i = offset; i < offset + nb_count; ++i)
    {
        if (m_edge_types[i] == input_edge_type)
        {
            type_offset = i;
            break;
        }
    }
    if (type_offset == std::numeric_limits<size_t>::max())
    {
        return std::nullopt;
    }
    const auto tp_count = m_edge_type_offset[type_offset + 1] - m_edge_type_offset[type_offset];
    auto fst = std::begin(m_edge_destination) + m_edge_type_offset[type_offset];
    auto lst = fst + tp_count;
    auto it = std::lower_bound(fst, lst, input_edge_dst);
    if (it == lst)
    {
        // Edge was not found in this partition.
        return std::nullopt;
    }
    return {it - std::begin(m_edge_destination)};
}

void Partition::GetEdgeFeature(uint64_t internal_src_node_id, NodeId input_edge_dst, Type input_edge_type,
                               std::optional<Timestamp> edge_ts, std::span<snark::FeatureMeta> features,
                               std::span<Timestamp> feature_flags, std::span<uint8_t> output) const
{
    auto file_ptr = m_edge_features->start();
    auto has_edge_features = EdgeFeatureOffset(internal_src_node_id, input_edge_dst, input_edge_type);
    if (!has_edge_features.has_value() || m_edge_feature_offset.empty() || m_edge_feature_index.empty())
    {
        return;
    }

    auto edge_offset = has_edge_features.value();
    auto feature_index_offset = m_edge_feature_offset[edge_offset];
    auto next_offset = m_edge_feature_offset[edge_offset + 1];
    auto curr = std::begin(output);

    for (size_t feature_index = 0; feature_index < features.size(); ++feature_index)
    {
        if ((!edge_ts.has_value() && feature_flags[feature_index] >= 0) ||
            (edge_ts.has_value() && feature_flags[feature_index] == edge_ts.value()))
        {
            curr += features[feature_index].second;
            continue;
        }

        const auto &feature = features[feature_index];
        const auto f_id = feature.first;
        const auto f_size = feature.second;

        // Requested feature_id is larger than known features, fill with 0s.
        if (next_offset - feature_index_offset <= uint64_t(f_id))
        {
            curr = std::fill_n(curr, f_size, 0);
            continue;
        }

        auto data_offset = m_edge_feature_index[feature_index_offset + f_id];
        auto stored_size = m_edge_feature_index[feature_index_offset + f_id + 1] - data_offset;
        Timestamp feature_ts = 0;

        if (edge_ts.has_value())
        {
            std::tie(data_offset, stored_size, feature_ts) =
                deserialize_temporal_feature(data_offset, stored_size, m_node_features, file_ptr, edge_ts.value());
        }

        feature_flags[feature_index] = feature_ts;
        curr = m_edge_features->read(data_offset, std::min<uint64_t>(f_size, stored_size), curr, file_ptr);
        if (stored_size < f_size)
        {
            curr = std::fill_n(curr, f_size - stored_size, 0);
        }
    }
}

void Partition::GetEdgeSparseFeature(uint64_t internal_src_node_id, NodeId input_edge_dst, Type input_edge_type,
                                     std::optional<Timestamp> edge_ts, std::span<const snark::FeatureId> features,
                                     std::span<Timestamp> feature_flags, int64_t prefix,
                                     std::span<int64_t> out_dimensions, std::vector<std::vector<int64_t>> &out_indices,
                                     std::vector<std::vector<uint8_t>> &out_values,
                                     std::vector<uint64_t> &values_sizes) const
{
    assert(features.size() == out_dimensions.size());
    auto file_ptr = m_edge_features->start();
    auto has_edge_features = EdgeFeatureOffset(internal_src_node_id, input_edge_dst, input_edge_type);
    if (!has_edge_features.has_value() || m_edge_feature_offset.empty() || m_edge_feature_index.empty())
    {
        return;
    }

    auto edge_offset = has_edge_features.value();
    auto feature_index_offset = m_edge_feature_offset[edge_offset];
    auto next_offset = m_edge_feature_offset[edge_offset + 1];
    for (size_t feature_index = 0; feature_index < features.size(); ++feature_index)
    {
        if ((!edge_ts.has_value() && feature_flags[feature_index] >= 0) ||
            (edge_ts.has_value() && feature_flags[feature_index] == edge_ts.value()))
        {
            continue;
        }

        const auto &feature = features[feature_index];
        // Requested feature_id is larger than known features, fill with 0s.
        if (next_offset - feature_index_offset <= uint64_t(feature))
        {
            continue;
        }

        auto data_offset = m_edge_feature_index[feature_index_offset + feature];
        auto stored_size = m_edge_feature_index[feature_index_offset + feature + 1] - data_offset;
        // Check if the feature is empty
        if (stored_size == 0)
        {
            continue;
        }

        Timestamp feature_ts = 0;
        if (edge_ts.has_value())
        {
            std::tie(data_offset, stored_size, feature_ts) =
                deserialize_temporal_feature(data_offset, stored_size, m_node_features, file_ptr, edge_ts.value());
        }

        feature_flags[feature_index] = feature_ts;
        deserialize_sparse_features(data_offset, stored_size, m_edge_features, feature, file_ptr, prefix,
                                    out_dimensions[feature_index], out_indices[feature_index],
                                    out_values[feature_index], values_sizes[feature_index]);
    }
}

void Partition::GetEdgeStringFeature(uint64_t internal_src_node_id, NodeId input_edge_dst, Type input_edge_type,
                                     std::optional<Timestamp> edge_ts, std::span<const snark::FeatureId> features,
                                     std::span<Timestamp> feature_flags, std::span<int64_t> out_dimensions,
                                     std::vector<uint8_t> &out_values) const
{
    assert(features.size() == out_dimensions.size());

    auto file_ptr = m_edge_features->start();
    auto has_edge_features = EdgeFeatureOffset(internal_src_node_id, input_edge_dst, input_edge_type);
    if (!has_edge_features.has_value())
    {
        return;
    }

    auto edge_offset = has_edge_features.value();
    auto feature_index_offset = m_edge_feature_offset[edge_offset];
    auto next_offset = m_edge_feature_offset[edge_offset + 1];

    for (size_t feature_index = 0; feature_index < features.size(); ++feature_index)
    {
        if ((!edge_ts.has_value() && feature_flags[feature_index] >= 0) ||
            (edge_ts.has_value() && feature_flags[feature_index] == edge_ts.value()))
        {
            continue;
        }

        const auto &feature = features[feature_index];
        // Requested feature_id is larger than known features, fill with 0s.
        if (next_offset - feature_index_offset <= uint64_t(feature))
        {
            continue;
        }

        auto data_offset = m_edge_feature_index[feature_index_offset + feature];
        auto stored_size = m_edge_feature_index[feature_index_offset + feature + 1] - data_offset;

        // Check if the feature is empty
        if (stored_size == 0)
        {
            continue;
        }

        Timestamp feature_ts = 0;
        if (edge_ts.has_value())
        {
            std::tie(data_offset, stored_size, feature_ts) =
                deserialize_temporal_feature(data_offset, stored_size, m_node_features, file_ptr, edge_ts.value());
        }

        feature_flags[feature_index] = feature_ts;
        deserialize_string_features(data_offset, stored_size, m_edge_features, file_ptr, out_dimensions[feature_index],
                                    out_values);
    }
}

void Partition::SampleNeighbor(int64_t seed, uint64_t internal_node_id, std::optional<Timestamp> node_ts,
                               std::span<const Type> in_edge_types, uint64_t count, std::span<NodeId> out_nodes,
                               std::span<Type> out_types, std::span<float> out_weights,
                               std::span<Timestamp> out_edge_created_ts, float &out_partition, NodeId default_node_id,
                               float default_weight, Type default_edge_type) const
{
    auto pos = 0;
    snark::Xoroshiro128PlusGenerator gen(seed);
    boost::random::uniform_real_distribution<float> real(0, 1.0f);

    const auto offset = m_neighbors_index[internal_node_id];
    const auto nb_count = m_neighbors_index[internal_node_id + 1] - offset;

    // Check if node doesn't have any neighbors
    if (nb_count == 0)
    {
        if (out_partition == 0)
        {
            std::fill_n(std::begin(out_nodes) + pos, count, default_node_id);
            std::fill_n(std::begin(out_types) + pos, count, default_edge_type);
            std::fill_n(std::begin(out_weights) + pos, count, default_weight);
            std::fill_n(std::begin(out_edge_created_ts) + pos, count, PLACEHOLDER_TIMESTAMP);
        }

        return;
    }

    float total_weight = 0;
    size_t curr_type = 0;
    const auto last_type = offset + nb_count;
    for (size_t i = offset; i < last_type; ++i)
    {
        for (; curr_type < in_edge_types.size() && in_edge_types[curr_type] < m_edge_types[i]; ++curr_type)
        {
        }
        if (curr_type == in_edge_types.size())
        {
            break;
        }
        for (; i < last_type && in_edge_types[curr_type] > m_edge_types[i]; ++i)
        {
        }
        if (i == last_type)
        {
            break;
        }

        if (m_edge_types[i] == in_edge_types[curr_type])
        {
            auto lst = m_edge_type_offset[i + 1];
            if (!node_ts.has_value())
            {
                total_weight += m_edge_weights[lst - 1];
                continue;
            }

            const auto fst = m_edge_type_offset[i];
            auto timestamp_span = std::span(m_edge_timestamps).subspan(fst, lst - fst);
            auto start_dist = fst;
            for (auto it = find_start(timestamp_span, node_ts.value());
                 it != std::end(timestamp_span) && !(it->second == -1 && it->first > node_ts.value());
                 it = find_start(timestamp_span, node_ts.value()))
            {
                auto diff = it - std::begin(timestamp_span);
                auto local_ts = timestamp_span.subspan(diff);
                auto last = find_last(local_ts, node_ts.value());
                const auto end_dist = size_t(last - std::begin(local_ts)) + start_dist + diff;
                total_weight += m_edge_weights[end_dist - 1];
                if (start_dist + diff != fst)
                {
                    total_weight -= m_edge_weights[start_dist - 1 + (it - std::begin(timestamp_span))];
                }

                if (last == std::begin(local_ts))
                {
                    ++last;
                }

                start_dist += diff + size_t(last - std::begin(local_ts));
                timestamp_span = timestamp_span.subspan(last - std::begin(local_ts) + diff);
            }
        }
    }

    out_partition += total_weight;
    if (total_weight == 0)
    {
        if (out_partition == 0)
        {
            std::fill_n(std::begin(out_nodes) + pos, count, default_node_id);
            std::fill_n(std::begin(out_types) + pos, count, default_edge_type);
            std::fill_n(std::begin(out_weights) + pos, count, default_weight);
            std::fill_n(std::begin(out_edge_created_ts) + pos, count, PLACEHOLDER_TIMESTAMP);
        }

        pos += count;
        return;
    }

    size_t left_over_neighbors = count;
    curr_type = 0;
    const auto overwrite_rate = total_weight / out_partition;
    for (size_t i = offset; i < offset + nb_count; ++i)
    {
        for (; curr_type < in_edge_types.size() && in_edge_types[curr_type] < m_edge_types[i]; ++curr_type)
        {
        }
        if (curr_type == in_edge_types.size())
        {
            break;
        }

        for (; i < last_type && in_edge_types[curr_type] > m_edge_types[i]; ++i)
        {
        }
        if (i == last_type)
        {
            break;
        }

        if (total_weight == 0 || left_over_neighbors == 0)
        {
            break;
        }

        if (m_edge_types[i] == in_edge_types[curr_type])
        {
            auto first = m_edge_type_offset[i];
            auto last = m_edge_type_offset[i + 1] - 1;
            if (node_ts.has_value())
            {
                ++last;
                auto timestamp_span = std::span(m_edge_timestamps).subspan(first, last - first);
                auto global_offset = first;
                for (auto it = find_start(timestamp_span, node_ts.value());
                     it != std::end(timestamp_span) && !(it->second == -1 && it->first > node_ts.value());
                     it = find_start(timestamp_span, node_ts.value()))
                {
                    const size_t it_dist = it - std::begin(timestamp_span);
                    const auto it_subspan = timestamp_span.subspan(it_dist);
                    auto lst = find_last(it_subspan, node_ts.value());
                    const size_t local_offset = global_offset + it_dist;
                    const auto end_dist = size_t(lst - std::begin(it_subspan)) + global_offset + it_dist;
                    auto type_weight = m_edge_weights[end_dist - 1];
                    if (local_offset != first)
                    {
                        type_weight -= m_edge_weights[local_offset - 1];
                    }

                    boost::random::binomial_distribution<int32_t> d(left_over_neighbors, type_weight / total_weight);
                    size_t type_count = type_weight == total_weight ? left_over_neighbors : d(gen);
                    total_weight -= type_weight;
                    for (size_t j = 0; j < type_count; ++j)
                    {
                        if (overwrite_rate < 1.0f && real(gen) > overwrite_rate)
                        {
                            continue;
                        }

                        float rnd = type_weight * real(gen);
                        if (local_offset != first)
                        {
                            rnd += m_edge_weights[local_offset - 1];
                        }

                        auto fst_nb = std::begin(m_edge_weights) + local_offset;
                        const auto curr_time_length =
                            (lst - std::begin(it_subspan)); // type safe equivalent of lst - it;
                        auto lst_nb = m_edge_weights.size() == local_offset + curr_time_length
                                          ? std::end(m_edge_weights)
                                          : std::begin(m_edge_weights) + local_offset + curr_time_length;
                        auto nb_pos = std::lower_bound(fst_nb, lst_nb, rnd);
                        size_t nb_offset = std::distance(fst_nb, nb_pos);
                        out_nodes[pos] = m_edge_destination[local_offset + nb_offset];
                        out_types[pos] = m_edge_types[i];
                        out_edge_created_ts[pos] = m_edge_timestamps[local_offset + nb_offset].first;
                        out_weights[pos] = (nb_offset == 0 && local_offset == first)
                                               ? m_edge_weights[local_offset]
                                               : m_edge_weights[local_offset + nb_offset] -
                                                     m_edge_weights[local_offset + nb_offset - 1];
                        ++pos;
                    }

                    left_over_neighbors -= type_count;
                    if (lst ==
                        std::begin(
                            it_subspan)) // cheking if positions are the same (lst == it) for different iterators.
                    {
                        ++lst;
                    }
                    const auto local_pos = size_t((lst - std::begin(it_subspan)) +
                                                  it_dist); // equivavlent of lst - std::begin(timestamp_span)
                    global_offset += local_pos;
                    timestamp_span = timestamp_span.subspan(local_pos);
                }
            }
            else
            {
                auto type_weight = m_edge_weights[last];
                boost::random::binomial_distribution<int32_t> d(left_over_neighbors, type_weight / total_weight);
                size_t type_count = type_weight == total_weight ? left_over_neighbors : d(gen);
                total_weight -= type_weight;
                for (size_t j = 0; j < type_count; ++j)
                {
                    if (overwrite_rate < 1.0f && real(gen) > overwrite_rate)
                    {
                        continue;
                    }

                    float rnd = type_weight * real(gen);
                    auto fst_nb = std::begin(m_edge_weights) + first;
                    auto lst_nb = m_edge_weights.size() == last ? std::end(m_edge_weights)
                                                                : std::begin(m_edge_weights) + last + 1;
                    auto nb_pos = std::lower_bound(fst_nb, lst_nb, rnd);
                    size_t nb_offset = std::distance(fst_nb, nb_pos);
                    out_nodes[pos] = m_edge_destination[first + nb_offset];
                    out_types[pos] = m_edge_types[i];
                    out_edge_created_ts[pos] = PLACEHOLDER_TIMESTAMP;
                    out_weights[pos] = nb_offset == 0
                                           ? m_edge_weights[first]
                                           : m_edge_weights[first + nb_offset] - m_edge_weights[first + nb_offset - 1];
                    ++pos;
                }

                left_over_neighbors -= type_count;
            }
        }
    }
}

// in_edge_types has to have types in strictly increasing order.
void Partition::UniformSampleNeighbor(bool without_replacement, uint64_t internal_node_id,
                                      std::optional<Timestamp> node_ts, std::span<const Type> in_edge_types,
                                      uint64_t count, std::span<NodeId> out_nodes, std::span<Type> out_types,
                                      std::span<Timestamp> out_edge_created_ts, uint64_t &out_partition_count,
                                      NodeId default_node_id, Type default_edge_type, AlgorithmL &sampler,
                                      WithReplacement &replacement_sampler) const
{
    if (without_replacement)
    {
        UniformSampleNeighborWithoutReplacement(internal_node_id, node_ts, in_edge_types, count, out_nodes, out_types,
                                                out_edge_created_ts, out_partition_count, default_node_id,
                                                default_edge_type, sampler);
    }
    else
    {
        UniformSampleNeighborWithReplacement(internal_node_id, node_ts, in_edge_types, count, out_nodes, out_types,
                                             out_edge_created_ts, out_partition_count, default_node_id,
                                             default_edge_type, replacement_sampler);
    }
}

// Advance neighbor_type_index(for edges coming out of the current node) and in_edge_type_index(requested edge
// types) until underlying types match.
bool advance_edge_types(size_t &in_edge_type_index, size_t &neighbor_type_index,
                        const std::span<const Type> &in_edge_types, const std::vector<Type> &neighbor_types,
                        size_t last_type)
{
    for (; in_edge_type_index < in_edge_types.size() &&
           in_edge_types[in_edge_type_index] < neighbor_types[neighbor_type_index];
         ++in_edge_type_index)
    {
    }
    if (in_edge_type_index == in_edge_types.size())
    {
        return false;
    }
    for (; neighbor_type_index < last_type && in_edge_types[in_edge_type_index] > neighbor_types[neighbor_type_index];
         ++neighbor_type_index)
    {
    }
    if (neighbor_type_index >= last_type)
    {
        return false;
    }

    return neighbor_types[neighbor_type_index] == in_edge_types[in_edge_type_index];
}

void Partition::UniformSampleNeighborWithReplacement(uint64_t internal_id, std::optional<Timestamp> node_ts,
                                                     std::span<const Type> in_edge_types, uint64_t count,
                                                     std::span<NodeId> out_nodes, std::span<Type> out_types,
                                                     std::span<Timestamp> out_edge_created_ts,
                                                     uint64_t &out_partition_count, NodeId default_node_id,
                                                     Type default_edge_type, WithReplacement &replacement_sampler) const
{
    const auto offset = m_neighbors_index[internal_id];
    const auto nb_count = m_neighbors_index[internal_id + 1] - offset;

    size_t in_edge_type_index = 0;
    const auto last_type = offset + nb_count;
    for (size_t neighbor_type_index = offset; neighbor_type_index < last_type; ++neighbor_type_index)
    {
        if (!advance_edge_types(in_edge_type_index, neighbor_type_index, in_edge_types, m_edge_types, last_type))
        {
            continue;
        }

        if (node_ts.has_value() && !m_edge_timestamps.empty())
        {
            auto first = m_edge_type_offset[neighbor_type_index];
            auto last = m_edge_type_offset[neighbor_type_index + 1] - 1;
            ++last;
            auto timestamp_span = std::span(m_edge_timestamps).subspan(first, last - first);
            auto global_offset = first;
            for (auto it = find_start(timestamp_span, node_ts.value());
                 it != std::end(timestamp_span) && !(it->second == -1 && it->first > node_ts.value());
                 it = find_start(timestamp_span, node_ts.value()))
            {
                const size_t it_dist = it - std::begin(timestamp_span);
                const auto it_subspan = timestamp_span.subspan(it_dist);
                auto lst = find_last(it_subspan, node_ts.value());
                auto stream = it_subspan.subspan(0, lst - std::begin(it_subspan));
                const size_t local_offset = global_offset + it_dist;

                const auto curr_weight = stream.size();
                out_partition_count += curr_weight;
                auto ts_span = std::span(m_edge_timestamps).subspan(local_offset, curr_weight);
                auto dst_span = std::span(m_edge_destination).subspan(local_offset, curr_weight);
                replacement_sampler.add(curr_weight, [out_nodes, out_types, out_edge_created_ts,
                                                      tp = m_edge_types[neighbor_type_index], ts_span, dst_span,
                                                      this](size_t pick, size_t offset) {
                    out_nodes[pick] = dst_span[offset];
                    out_types[pick] = tp;
                    out_edge_created_ts[pick] =
                        m_edge_timestamps.empty() ? PLACEHOLDER_TIMESTAMP : ts_span[offset].first;
                });

                if (lst ==
                    std::begin(it_subspan)) // cheking if positions are the same (lst == it) for different iterators.
                {
                    ++lst;
                }
                const auto local_pos =
                    size_t((lst - std::begin(it_subspan)) + it_dist); // equivavlent of lst - std::begin(timestamp_span)
                global_offset += local_pos;
                timestamp_span = timestamp_span.subspan(local_pos);
            }
        }
        else
        {
            const auto curr_weight =
                m_edge_type_offset[neighbor_type_index + 1] - m_edge_type_offset[neighbor_type_index];
            out_partition_count += curr_weight;
            const size_t local_offset = m_edge_type_offset[neighbor_type_index];
            auto dst_span = std::span(m_edge_destination).subspan(local_offset, curr_weight);
            replacement_sampler.add(curr_weight,
                                    [out_nodes, out_types, out_edge_created_ts, tp = m_edge_types[neighbor_type_index],
                                     dst_span, this](size_t pick, size_t offset) {
                                        out_nodes[pick] = dst_span[offset];
                                        out_types[pick] = tp;
                                        out_edge_created_ts[pick] = PLACEHOLDER_TIMESTAMP;
                                    });
        }
    }
}

void Partition::UniformSampleNeighborWithoutReplacement(uint64_t internal_id, std::optional<Timestamp> node_ts,
                                                        std::span<const Type> in_edge_types, uint64_t count,
                                                        std::span<NodeId> out_nodes, std::span<Type> out_types,
                                                        std::span<Timestamp> out_edge_created_ts,
                                                        uint64_t &out_partition_count, NodeId default_node_id,
                                                        Type default_edge_type, AlgorithmL &sampler) const
{
    const auto offset = m_neighbors_index[internal_id];
    const auto nb_count = m_neighbors_index[internal_id + 1] - offset;

    size_t curr_type = 0;
    const auto last_type = offset + nb_count;

    for (size_t i = offset; i < last_type; ++i)
    {
        if (!advance_edge_types(curr_type, i, in_edge_types, m_edge_types, last_type))
        {
            continue;
        }

        if (node_ts.has_value() && !m_edge_timestamps.empty())
        {
            auto first = m_edge_type_offset[i];
            auto last = m_edge_type_offset[i + 1] - 1;
            ++last;
            auto timestamp_span = std::span(m_edge_timestamps).subspan(first, last - first);
            for (auto it = find_start(timestamp_span, node_ts.value());
                 it != std::end(timestamp_span) && !(it->second == -1 && it->first > node_ts.value());
                 it = find_start(timestamp_span, node_ts.value()))
            {
                const size_t it_dist = it - std::begin(timestamp_span);
                const auto it_subspan = timestamp_span.subspan(it_dist);
                auto lst = find_last(it_subspan, node_ts.value());
                auto stream = it_subspan.subspan(0, lst - std::begin(it_subspan));

                const auto stream_size = stream.size();
                out_partition_count += stream_size;
                auto dest_span = std::span(m_edge_destination).subspan(first + it_dist, stream_size);
                auto ts_span = std::span(m_edge_timestamps).subspan(first + it_dist, stream_size);
                sampler.add(stream_size, [&dest_span, &out_nodes, &out_types, &out_edge_created_ts, ts_span,
                                          tp = in_edge_types[curr_type]](size_t pick, size_t offset) {
                    out_nodes[pick] = dest_span[offset];
                    out_types[pick] = tp;
                    out_edge_created_ts[pick] = ts_span[offset].first;
                });

                if (lst ==
                    std::begin(it_subspan)) // cheking if positions are the same (lst == it) for different iterators.
                {
                    ++lst;
                }
                const auto local_pos =
                    size_t((lst - std::begin(it_subspan)) + it_dist); // equivavlent of lst - std::begin(timestamp_span)
                timestamp_span = timestamp_span.subspan(local_pos);
            }
        }
        else
        {
            auto first = m_edge_type_offset[i];
            auto last = m_edge_type_offset[i + 1];
            const auto stream_size = last - first;
            out_partition_count += stream_size;
            auto dest_span = std::span(m_edge_destination).subspan(first, stream_size);
            sampler.add(dest_span.size(), [&dest_span, &out_nodes, &out_types,
                                           tp = in_edge_types[curr_type]](size_t pick, size_t offset) {
                out_nodes[pick] = dest_span[offset];
                out_types[pick] = tp;
            });
        }
    }
}

Metadata Partition::GetMetadata() const
{
    return m_metadata;
}
} // namespace snark
