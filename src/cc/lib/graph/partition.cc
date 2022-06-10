// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstring>
#include <limits>

#include "locator.h"
#include "partition.h"
#include "sampler.h"

#include "boost/random/binomial_distribution.hpp"
#include "boost/random/uniform_real_distribution.hpp"
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
} // namespace

Partition::Partition(std::filesystem::path path, std::string suffix, PartitionStorageType storage_type)
    : m_metadata(path), m_storage_type(storage_type)
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
        auto full_path = path / ("edge_features" + suffix + ".data");
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

void Partition::GetNodeFeature(uint64_t internal_id, std::span<snark::FeatureMeta> features,
                               std::span<uint8_t> output) const
{
    auto file_ptr = m_node_features->start();
    auto curr = std::begin(output);

    auto feature_index_offset = m_node_index[internal_id];
    auto next_offset = m_node_index[internal_id + 1];

    for (const auto &feature : features)
    {
        const auto feature_id = feature.first;
        const auto feature_size = feature.second;

        // Requested feature_id is larger than known features, fill with 0s.
        if (next_offset - feature_index_offset <= uint64_t(feature_id) || m_node_feature_index.empty())
        {
            curr = std::fill_n(curr, feature_size, 0);
            continue;
        }

        const auto data_offset = m_node_feature_index[feature_index_offset + feature_id];
        const auto stored_size = m_node_feature_index[feature_index_offset + feature_id + 1] - data_offset;
        curr = m_node_features->read(data_offset, std::min<uint64_t>(feature_size, stored_size), curr, file_ptr);
        if (stored_size < feature_size)
        {
            curr = std::fill_n(curr, feature_size - stored_size, 0);
        }
    }
}

void Partition::GetNodeSparseFeature(uint64_t internal_node_id, std::span<const snark::FeatureId> features,
                                     int64_t prefix, std::span<int64_t> out_dimensions,
                                     std::vector<std::vector<int64_t>> &out_indices,
                                     std::vector<std::vector<uint8_t>> &out_values) const
{
    assert(features.size() == out_dimensions.size());
    auto file_ptr = m_node_features->start();

    auto feature_index_offset = m_node_index[internal_node_id];
    auto next_offset = m_node_index[internal_node_id + 1];

    for (size_t feature_index = 0; feature_index < features.size(); ++feature_index)
    {
        const auto feature = features[feature_index];
        // Requested feature_id is larger than known features, skip.
        if (next_offset - feature_index_offset <= uint64_t(feature) || m_node_feature_index.empty())
        {
            continue;
        }

        const auto data_offset = m_node_feature_index[feature_index_offset + feature];
        const auto stored_size = m_node_feature_index[feature_index_offset + feature + 1] - data_offset;
        // Check if the feature is empty
        if (stored_size == 0)
        {
            continue;
        }
        assert(stored_size > 12); // minimum is 4 bytes to record there is a single index, actual index (8 bytes)
                                  // and some data(>0 bytes).
        uint32_t indices_size = 0;
        auto indices_size_output = std::span(reinterpret_cast<uint8_t *>(&indices_size), 4);
        m_node_features->read(data_offset, indices_size_output.size(), std::begin(indices_size_output), file_ptr);
        uint32_t indices_dim = 0;
        auto indices_dim_output = std::span(reinterpret_cast<uint8_t *>(&indices_dim), 4);
        m_node_features->read(data_offset + 4, indices_dim_output.size(), std::begin(indices_dim_output), file_ptr);
        out_dimensions[feature_index] = int64_t(indices_dim);

        assert(indices_size % indices_dim == 0);
        size_t num_values = indices_size / indices_dim;
        const auto old_len = out_indices[feature_index].size();
        out_indices[feature_index].resize(old_len + indices_size + num_values, prefix);
        auto output = std::span(reinterpret_cast<uint8_t *>(out_indices[feature_index].data()) + old_len * 8,
                                (indices_size + num_values) * 8);

        // Read expected feature_dim indices in bytes
        size_t indices_offset = data_offset + 8;
        auto curr = std::begin(output);
        for (size_t i = 0; i < num_values; ++i)
        {
            curr += 8;
            curr = m_node_features->read(indices_offset, indices_dim * 8, curr, file_ptr);
            indices_offset += 8 * indices_dim;
        }

        // Read values
        const auto values_length = stored_size - indices_size * 8 - 8;
        const auto old_values_length = out_values[feature_index].size();
        out_values[feature_index].resize(old_values_length + values_length);
        auto out_values_span = std::span(out_values[feature_index]).subspan(old_values_length);
        m_node_features->read(indices_offset, values_length, std::begin(out_values_span), file_ptr);
    }
}

void Partition::GetNodeStringFeature(uint64_t internal_node_id, std::span<const snark::FeatureId> features,
                                     std::span<int64_t> out_dimensions, std::vector<uint8_t> &out_values) const
{
    assert(features.size() == out_dimensions.size());
    auto file_ptr = m_node_features->start();

    auto feature_index_offset = m_node_index[internal_node_id];
    auto next_offset = m_node_index[internal_node_id + 1];

    for (size_t feature_index = 0; feature_index < features.size(); ++feature_index)
    {
        const auto feature = features[feature_index];
        // Requested feature_id is larger than known features, skip.
        if (next_offset - feature_index_offset <= uint64_t(feature) || m_node_feature_index.empty())
        {
            continue;
        }

        const auto data_offset = m_node_feature_index[feature_index_offset + feature];
        const auto stored_size = m_node_feature_index[feature_index_offset + feature + 1] - data_offset;
        // Check if the feature is empty
        if (stored_size == 0)
        {
            continue;
        }

        out_dimensions[feature_index] = stored_size;
        const auto old_values_length = out_values.size();
        out_values.resize(old_values_length + stored_size);
        auto out_values_span = std::span(out_values).subspan(old_values_length);
        m_node_features->read(data_offset, stored_size, std::begin(out_values_span), file_ptr);
    }
}

template<class F>
size_t Partition::FetchNeighborInfo(uint64_t internal_id, std::span<const Type> edge_types, F func) const

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
        if (m_edge_types[i] == edge_types[curr_type])
        {
            const auto start = m_edge_type_offset[i];
            const auto last = m_edge_type_offset[i + 1];

            result += func(start, last, i);
        }
    }
    return result;
}


size_t Partition::NeighborCount(uint64_t internal_id, std::span<const Type> edge_types) const
{   
    auto lambda = [&](auto start, auto last, int i)
                        {
                            return last - start;
                        };

    return FetchNeighborInfo(internal_id, edge_types, lambda);
}

size_t Partition::FullNeighbor(uint64_t internal_id, std::span<const Type> edge_types,
                               std::vector<NodeId> &out_neighbors_ids, std::vector<Type> &out_edge_types,
                               std::vector<float> &out_edge_weights) const
{   
    auto lambda = [&](auto start, auto last, int i)
            {
                // m_edge_destination[last-1]+1 - take the last element and then advance the pointer
                // to imitate std::end, otherwise we'll have an out of range exception.
                out_neighbors_ids.insert(std::end(out_neighbors_ids), &m_edge_destination[start],
                                                &m_edge_destination[last - 1] + 1);
                auto original_type_size = out_edge_types.size();
                out_edge_types.resize(original_type_size + last - start, m_edge_types[i]);
                out_edge_weights.reserve(out_edge_weights.size() + last - start);
                for (size_t index = start; index < last; ++index)
                {
                    out_edge_weights.emplace_back(index > start ? m_edge_weights[index] - m_edge_weights[index - 1]
                                                                        : m_edge_weights[start]);
                }

                return last - start;
            };

    return FetchNeighborInfo(internal_id, edge_types, lambda);
}

bool Partition::GetEdgeFeature(uint64_t internal_src_node_id, NodeId input_edge_dst, Type input_edge_type,
                               std::span<snark::FeatureMeta> features, std::span<uint8_t> output) const
{
    auto file_ptr = m_edge_features->start();
    auto curr = std::begin(output);

    const auto offset = m_neighbors_index[internal_src_node_id];
    const auto nb_count = m_neighbors_index[internal_src_node_id + 1] - offset;

    // Check if node doesn't have any neighbors
    if (nb_count == 0)
    {
        return false;
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
        return false;
    }
    const auto tp_count = m_edge_type_offset[type_offset + 1] - m_edge_type_offset[type_offset];
    auto fst = std::begin(m_edge_destination) + m_edge_type_offset[type_offset];
    auto lst = fst + tp_count;
    auto it = std::lower_bound(fst, lst, input_edge_dst);
    if (it == lst)
    {
        // Edge was not found in this partition.
        return false;
    }
    if (m_edge_feature_offset.empty() || m_edge_feature_index.empty())
    {
        std::fill(std::begin(output), std::end(output), 0);
        return true;
    }

    auto edge_offset = it - std::begin(m_edge_destination);
    auto feature_index_offset = m_edge_feature_offset[edge_offset];
    auto next_offset = m_edge_feature_offset[edge_offset + 1];

    for (const auto &feature : features)
    {
        const auto f_id = feature.first;
        const auto f_size = feature.second;

        // Requested feature_id is larger than known features, fill with 0s.
        if (next_offset - feature_index_offset <= uint64_t(f_id))
        {
            curr = std::fill_n(curr, f_size, 0);
            continue;
        }

        const auto data_offset = m_edge_feature_index[feature_index_offset + f_id];
        const auto stored_size = m_edge_feature_index[feature_index_offset + f_id + 1] - data_offset;
        curr = m_edge_features->read(data_offset, std::min<uint64_t>(f_size, stored_size), curr, file_ptr);
        if (stored_size < f_size)
        {
            const auto f_id = feature.first;
            const auto f_size = feature.second;

            // Requested feature_id is larger than known features, fill with 0s.
            if (next_offset - feature_index_offset <= uint64_t(f_id))
            {
                curr = std::fill_n(curr, f_size, 0);
                continue;
            }

            const auto data_offset = m_edge_feature_index[feature_index_offset + f_id];
            const auto stored_size = m_edge_feature_index[feature_index_offset + f_id + 1] - data_offset;

            curr = m_edge_features->read(data_offset, std::min<uint64_t>(f_size, stored_size), curr, file_ptr);
            if (stored_size < f_size)
            {
                curr = std::fill_n(curr, f_size - stored_size, 0);
            }
        }
    }

    return true;
}

bool Partition::GetEdgeSparseFeature(uint64_t internal_src_node_id, NodeId input_edge_dst, Type input_edge_type,
                                     std::span<const snark::FeatureId> features, int64_t prefix,
                                     std::span<int64_t> out_dimensions, std::vector<std::vector<int64_t>> &out_indices,
                                     std::vector<std::vector<uint8_t>> &out_values) const
{
    assert(features.size() == out_dimensions.size());

    auto file_ptr = m_edge_features->start();
    const auto offset = m_neighbors_index[internal_src_node_id];
    const auto nb_count = m_neighbors_index[internal_src_node_id + 1] - offset;

    // Check if node doesn't have any neighbors
    if (nb_count == 0)
    {
        return false;
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
        return false;
    }
    const auto tp_count = m_edge_type_offset[type_offset + 1] - m_edge_type_offset[type_offset];
    auto fst = std::begin(m_edge_destination) + m_edge_type_offset[type_offset];
    auto lst = fst + tp_count;
    auto it = std::lower_bound(fst, lst, input_edge_dst);
    if (it == lst)
    {
        // Edge was not found in this partition.
        return false;
    }
    if (m_edge_feature_offset.empty() || m_edge_feature_index.empty())
    {
        return true;
    }

    auto edge_offset = it - std::begin(m_edge_destination);
    auto feature_index_offset = m_edge_feature_offset[edge_offset];
    auto next_offset = m_edge_feature_offset[edge_offset + 1];

    for (size_t feature_index = 0; feature_index < features.size(); ++feature_index)
    {
        const auto feature = features[feature_index];
        // Requested feature_id is larger than known features, fill with 0s.
        if (next_offset - feature_index_offset <= uint64_t(feature))
        {
            continue;
        }

        const auto data_offset = m_edge_feature_index[feature_index_offset + feature];
        const auto stored_size = m_edge_feature_index[feature_index_offset + feature + 1] - data_offset;

        // Check if the feature is empty
        if (stored_size == 0)
        {
            continue;
        }

        assert(stored_size > 12); // minimum is 4 bytes to record there is a single index, actual index (8 bytes)
                                  // and some data(>0 bytes).
        uint32_t indices_size = 0;
        auto indices_size_output = std::span(reinterpret_cast<uint8_t *>(&indices_size), 4);
        m_edge_features->read(data_offset, indices_size_output.size(), std::begin(indices_size_output), file_ptr);

        uint32_t indices_dim = 0;
        auto indices_dim_output = std::span(reinterpret_cast<uint8_t *>(&indices_dim), 4);
        m_edge_features->read(data_offset + 4, indices_dim_output.size(), std::begin(indices_dim_output), file_ptr);
        out_dimensions[feature_index] = int64_t(indices_dim);

        assert(indices_size % indices_dim == 0);
        size_t num_values = indices_size / indices_dim;
        const auto old_len = out_indices[feature_index].size();
        out_indices[feature_index].resize(old_len + indices_size + num_values, prefix);
        auto output = std::span(reinterpret_cast<uint8_t *>(out_indices[feature_index].data()) + old_len * 8,
                                (indices_size + num_values) * 8);

        // Read expected feature_dim indices in bytes
        size_t indices_offset = data_offset + 8;
        auto curr = std::begin(output);
        for (size_t i = 0; i < num_values; ++i)
        {
            curr += 8;
            curr = m_edge_features->read(indices_offset, indices_dim * 8, curr, file_ptr);
            indices_offset += 8 * indices_dim;
        }

        // Read values
        const auto values_length = stored_size - indices_size * 8 - 8;
        const auto old_values_length = out_values[feature_index].size();
        out_values[feature_index].resize(old_values_length + values_length);
        auto out_values_span = std::span(out_values[feature_index]).subspan(old_values_length);
        m_edge_features->read(indices_offset, values_length, std::begin(out_values_span), file_ptr);
    }

    return true;
}

bool Partition::GetEdgeStringFeature(uint64_t internal_src_node_id, NodeId input_edge_dst, Type input_edge_type,
                                     std::span<const snark::FeatureId> features, std::span<int64_t> out_dimensions,
                                     std::vector<uint8_t> &out_values) const
{
    assert(features.size() == out_dimensions.size());

    auto file_ptr = m_edge_features->start();
    const auto offset = m_neighbors_index[internal_src_node_id];
    const auto nb_count = m_neighbors_index[internal_src_node_id + 1] - offset;

    // Check if node doesn't have any neighbors
    if (nb_count == 0)
    {
        return false;
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
        return false;
    }
    const auto tp_count = m_edge_type_offset[type_offset + 1] - m_edge_type_offset[type_offset];
    auto fst = std::begin(m_edge_destination) + m_edge_type_offset[type_offset];
    auto lst = fst + tp_count;
    auto it = std::lower_bound(fst, lst, input_edge_dst);
    if (it == lst)
    {
        // Edge was not found in this partition.
        return false;
    }
    if (m_edge_feature_offset.empty() || m_edge_feature_index.empty())
    {
        return true;
    }

    auto edge_offset = it - std::begin(m_edge_destination);
    auto feature_index_offset = m_edge_feature_offset[edge_offset];
    auto next_offset = m_edge_feature_offset[edge_offset + 1];

    for (size_t feature_index = 0; feature_index < features.size(); ++feature_index)
    {
        const auto feature = features[feature_index];
        // Requested feature_id is larger than known features, fill with 0s.
        if (next_offset - feature_index_offset <= uint64_t(feature))
        {
            continue;
        }

        const auto data_offset = m_edge_feature_index[feature_index_offset + feature];
        const auto stored_size = m_edge_feature_index[feature_index_offset + feature + 1] - data_offset;

        // Check if the feature is empty
        if (stored_size == 0)
        {
            continue;
        }

        out_dimensions[feature_index] = stored_size;
        const auto old_values_length = out_values.size();
        out_values.resize(old_values_length + stored_size);
        auto out_values_span = std::span(out_values).subspan(old_values_length);
        m_edge_features->read(data_offset, stored_size, std::begin(out_values_span), file_ptr);
    }

    return true;
}

void Partition::SampleNeighbor(int64_t seed, uint64_t internal_node_id, std::span<const Type> in_edge_types,
                               uint64_t count, std::span<NodeId> out_nodes, std::span<Type> out_types,
                               std::span<float> out_weights, float &out_partition, NodeId default_node_id,
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
        }

        pos += count;
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
            auto last = m_edge_type_offset[i + 1] - 1;
            total_weight += m_edge_weights[last];
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
            const auto first = m_edge_type_offset[i];
            const auto last = m_edge_type_offset[i + 1] - 1;
            const auto type_weight = m_edge_weights[last];

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
                auto lst_nb =
                    m_edge_weights.size() == last ? std::end(m_edge_weights) : std::begin(m_edge_weights) + last + 1;
                auto nb_pos = std::lower_bound(fst_nb, lst_nb, rnd);
                size_t nb_offset = std::distance(fst_nb, nb_pos);
                out_nodes[pos] = m_edge_destination[first + nb_offset];
                out_types[pos] = m_edge_types[i];
                out_weights[pos] = nb_offset == 0
                                       ? m_edge_weights[first]
                                       : m_edge_weights[first + nb_offset] - m_edge_weights[first + nb_offset - 1];
                ++pos;
            }
            left_over_neighbors -= type_count;
        }
    }
}

// in_edge_types has to have types in strictly increasing order.
void Partition::UniformSampleNeighbor(bool without_replacement, int64_t seed, uint64_t internal_node_id,
                                      std::span<const Type> in_edge_types, uint64_t count, std::span<NodeId> out_nodes,
                                      std::span<Type> out_types, uint64_t &out_partition_count, NodeId default_node_id,
                                      Type default_edge_type) const
{
    if (without_replacement)
    {
        UniformSampleNeighborWithoutReplacement(seed, internal_node_id, in_edge_types, count, out_nodes, out_types,
                                                out_partition_count, default_node_id, default_edge_type);
    }
    else
    {
        UniformSampleNeighborWithReplacement(seed, internal_node_id, in_edge_types, count, out_nodes, out_types,
                                             out_partition_count, default_node_id, default_edge_type);
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

void Partition::UniformSampleNeighborWithReplacement(int64_t seed, uint64_t internal_id,
                                                     std::span<const Type> in_edge_types, uint64_t count,
                                                     std::span<NodeId> out_nodes, std::span<Type> out_types,
                                                     uint64_t &out_partition_count, NodeId default_node_id,
                                                     Type default_edge_type) const
{
    size_t pos = 0;
    // It is important to use a good generator, because we use it to pick a number and merge results from multiple
    // partitions. E.g. rand_48 engine will produce correlated samples.
    snark::Xoroshiro128PlusGenerator gen(seed);
    boost::random::uniform_real_distribution<float> toss(0, 1);

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

        const auto curr_weight = m_edge_type_offset[neighbor_type_index + 1] - m_edge_type_offset[neighbor_type_index];
        out_partition_count += curr_weight;
        // Probabilities to select correct types will converge to right values:
        // E.g. we have 3 neighbor types with 5, 9 and 11 elements, then probability
        // to the first neighbor to have type 0 is 1 *(5/14) *(14/25) = 5/25
        const auto merge_rate = float(curr_weight) / out_partition_count;
        for (size_t nb = 0; nb < count; ++nb)
        {
            if (merge_rate == 1.0f || toss(gen) < merge_rate)
            {
                size_t pick = toss(gen) * curr_weight;
                out_nodes[pos + nb] = m_edge_destination[m_edge_type_offset[neighbor_type_index] + pick];
                out_types[pos + nb] = m_edge_types[neighbor_type_index];
            }
        }
    }

    if (out_partition_count == 0)
    {
        std::fill_n(std::begin(out_nodes) + pos, count, default_node_id);
        std::fill_n(std::begin(out_types) + pos, count, default_edge_type);
    }

    pos += count;
}

// Sample min(partition_weight, count) neighbors from a range (0..partition_weight) using reservoir sampling
// and store indices in the `interim_neighbors`.
void contiguous_uniform_sample_helper(size_t partition_weight, uint64_t count, std::vector<size_t> &interim_neighbors,
                                      boost::random::uniform_real_distribution<double> &toss,
                                      snark::Xoroshiro128PlusGenerator &gen)
{
    if (partition_weight <= count)
    {
        size_t start = 0;
        std::generate_n(std::back_inserter(interim_neighbors), partition_weight, [&start]() {
            const auto result = start;
            ++start;
            return result;
        });
        return;
    }

    for (size_t node = 0; node < count; ++node)
    {
        interim_neighbors.emplace_back(node);
    }

    float w = std::exp(std::log(toss(gen)) / count);
    size_t i = count - 1;
    while (i < partition_weight)
    {
        i += std::floor(std::log(toss(gen)) / std::log(1 - w)) + 1;
        if (i < partition_weight)
        {
            const size_t pick = toss(gen) * count;
            interim_neighbors[pick] = i;
            w = w * std::exp(std::log(toss(gen)) / count);
        }
    }
}

// We can't use bernoulli sampling here, because the total number of actual neighbors in each list might be less
// then `count` and we should rather do a direct sampling from both lists and track their overall length.
// E.g. [1, 2, default, default, default] merged with [3,4, default, default, default] should be [1, 2, 3, 4,
// default], rather than something like [1, 4, default, default, default].
// Merge procedure is following:
// 1. Pick a list with a probability proportional to the length of this list(without default elements).
// 2. Randomly pick an element from the list and put it in the result.
// 3. To make sure there are no repeating elements(without replacement), put selected element in the head of the
// list and track the tail for selection.
// 4. Assign the result array weight equal to the sum of lengths of original lists to make sure later merge will be
// proportional. To show this procedure is not biased, lets merge lists [1,2,3, default] and [4, 5, 6, 7] into one
// list with 4 elements. Probability that element 1 will be selected first is (3/7)*(1/3) = 1/7, same as element 5:
// (4/7) * (1/4) = 1/7
void Partition::UniformSampleMergeWithoutReplacement(
    uint64_t count, std::vector<NodeId> &left_neighbors, std::vector<Type> &left_types, uint64_t left_weight,
    std::vector<size_t> &interim_neighbors, std::vector<size_t> &type_counts, std::vector<Type> &type_values,
    std::vector<size_t> &destination_offsets, uint64_t right_weight, std::span<NodeId> out_neighbors,
    std::span<Type> out_edge_types, NodeId default_node_id, Type default_edge_type,
    boost::random::uniform_real_distribution<double> &toss, snark::Xoroshiro128PlusGenerator &gen) const
{
    size_t left_max = std::min(count, left_weight);
    size_t left_pos = 0;
    size_t right_max = std::min(count, right_weight);
    size_t right_pos = 0;

    size_t out_pos = 0;
    for (; out_pos < count && (left_weight + right_weight > 0); ++out_pos)
    {
        const auto merge_rate = float(left_weight) / (left_weight + right_weight);
        if (left_pos < left_max && toss(gen) < merge_rate)
        {
            size_t pick = size_t(toss(gen) * (left_max - left_pos)) + left_pos;
            std::swap(left_neighbors[pick], left_neighbors[left_pos]);
            std::swap(left_types[pick], left_types[left_pos]);
            out_neighbors[out_pos] = left_neighbors[left_pos];
            out_edge_types[out_pos] = left_types[left_pos];
            ++left_pos;
            --left_weight;
        }
        else if (right_pos < right_max)
        {
            // It is important to do conversion to size_t first, because if a random generator
            // produces 0.999994, then result might be equal right_max, because float doesn't have
            // enough precision(e.g. 0.99994*(24-9)+9 = 24 in floats vs 23 in size_t).
            size_t pick = size_t(toss(gen) * (right_max - right_pos)) + right_pos;
            std::swap(interim_neighbors[pick], interim_neighbors[right_pos]);

            // Recover destination id: first determine edge type and then use destination offset to find the
            // node_id. For example if the total number of neighbors is 5 and number of edge types is 2 and 3. This
            // means the type counts vector will store accumulated counts [2, 5]. So when we sample indices 1 and 4,
            // then we'll use std::lower_bound to find the edge types are 0 and 1 respectively. First edge
            // destination will use offset 1-0=1 in the m_edge_destination array for a given edge type and second
            // edge offset will be 4-2=2 for type 1.
            size_t type_offset =
                std::lower_bound(std::begin(type_counts), std::end(type_counts), interim_neighbors[right_pos] + 1) -
                std::begin(type_counts);
            out_edge_types[out_pos] = type_values[type_offset];
            size_t prev_type = type_offset == 0 ? 0 : type_counts[type_offset - 1];
            out_neighbors[out_pos] =
                m_edge_destination[destination_offsets[type_offset] + interim_neighbors[right_pos] - prev_type];
            ++right_pos;
            --right_weight;
        }
        else
        {
            --out_pos;
        }
    }
    for (; out_pos < count; ++out_pos)
    {
        out_neighbors[out_pos] = default_node_id;
        out_edge_types[out_pos] = default_edge_type;
    }
}

void Partition::UniformSampleNeighborWithoutReplacement(int64_t seed, uint64_t internal_id,
                                                        std::span<const Type> in_edge_types, uint64_t count,
                                                        std::span<NodeId> out_nodes, std::span<Type> out_types,
                                                        uint64_t &out_partition_count, NodeId default_node_id,
                                                        Type default_edge_type) const
{
    size_t pos = 0;
    snark::Xoroshiro128PlusGenerator gen(seed);
    boost::random::uniform_real_distribution<double> toss(0, 1);

    // Temporary variables, allocate memory once and reuse it for every node.
    std::vector<size_t> type_counts;
    type_counts.reserve(in_edge_types.size());
    std::vector<Type> type_values;
    type_values.reserve(in_edge_types.size());
    std::vector<size_t> destination_offsets;
    destination_offsets.reserve(in_edge_types.size());
    std::vector<size_t> interim_neighbors;
    interim_neighbors.reserve(count);
    std::vector<NodeId> prev_nodes;
    prev_nodes.reserve(count);
    std::vector<Type> prev_types;
    prev_types.reserve(count);

    const auto offset = m_neighbors_index[internal_id];
    const auto nb_count = m_neighbors_index[internal_id + 1] - offset;

    size_t curr_type = 0;
    const auto last_type = offset + nb_count;

    // In order to avoid storing all node neighbors we'll find the total number of neighbors for given types
    // and then sample from a continuous range of elements from 1..#neighbors.
    // We store `type_counts`, `type_values` and `destination_offsets` to recover destination node ids later
    // by sampled indices with O(log(#edge_types)) complexity.
    type_counts.clear();
    type_values.clear();
    destination_offsets.clear();
    interim_neighbors.clear();
    size_t partition_weight = 0;
    for (size_t i = offset; i < last_type; ++i)
    {
        if (!advance_edge_types(curr_type, i, in_edge_types, m_edge_types, last_type))
        {
            continue;
        }

        const auto curr_weight = m_edge_type_offset[i + 1] - m_edge_type_offset[i];
        partition_weight += curr_weight;
        type_counts.emplace_back(partition_weight);
        type_values.emplace_back(m_edge_types[i]);
        destination_offsets.emplace_back(m_edge_type_offset[i]);
    }

    contiguous_uniform_sample_helper(partition_weight, count, interim_neighbors, toss, gen);

    size_t prev_max = std::min(count, out_partition_count);
    prev_nodes.assign(std::begin(out_nodes) + pos, std::begin(out_nodes) + pos + prev_max);
    prev_types.assign(std::begin(out_types) + pos, std::begin(out_types) + pos + prev_max);

    UniformSampleMergeWithoutReplacement(count, prev_nodes, prev_types, out_partition_count, interim_neighbors,
                                         type_counts, type_values, destination_offsets, partition_weight,
                                         out_nodes.subspan(pos, count), out_types.subspan(pos, count), default_node_id,
                                         default_edge_type, toss, gen);

    pos += count;
    out_partition_count += partition_weight;
}

Metadata Partition::GetMetadata() const
{
    return m_metadata;
}
} // namespace snark
