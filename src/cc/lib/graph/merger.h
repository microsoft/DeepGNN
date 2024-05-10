// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#ifndef SNARK_MERGER_H
#define SNARK_MERGER_H

#include <atomic>
#include <cassert>
#include <mutex>
#include <span>
#include <vector>

#include "src/cc/lib/graph/logger.h"
#include "src/cc/lib/graph/types.h"

namespace snark
{

// Index to look up feature coordinates to return them in sorted order.
// shard, index offset, index count, value offset, value count
struct SparseFeatureIndex
{
    int shard;
    int index_offset;
    int index_count;
    int value_offset;
    int value_count;
    snark::Timestamp timestamp;
};

template <typename SparseFeaturesReply>
void ExtractFeatures(const std::vector<SparseFeatureIndex> &response_index,
                     const std::vector<SparseFeaturesReply> &replies, std::span<int64_t> out_dimensions,
                     std::vector<std::vector<int64_t>> &out_indices, std::vector<std::vector<uint8_t>> &out_values,
                     size_t node_count)
{
    const size_t feature_count = out_dimensions.size();
    for (size_t feature_index = 0; feature_index < feature_count; ++feature_index)
    {
        for (size_t node_index = 0; node_index < node_count; ++node_index)
        {
            auto &response_index_item = response_index[node_index * feature_count + feature_index];
            if (response_index_item.shard < 0 || replies[response_index_item.shard].indices().empty())
            {
                continue;
            }

            std::copy_n(std::begin(replies[response_index_item.shard].indices()) + response_index_item.index_offset,
                        response_index_item.index_count, std::back_inserter(out_indices[feature_index]));
            std::copy_n(std::begin(replies[response_index_item.shard].values()) + response_index_item.value_offset,
                        response_index_item.value_count, std::back_inserter(out_values[feature_index]));
        }
    }
}

template <typename StringFeaturesReply>
void ExtractStringFeatures(const std::vector<std::tuple<size_t, size_t, snark::Timestamp>> &response_index,
                           const std::vector<StringFeaturesReply> &replies, std::span<int64_t> dimensions,
                           std::vector<uint8_t> &out_values)
{
    size_t feature_index = 0;
    for (const auto &index : response_index)
    {
        const auto response_id = std::get<0>(index);
        const auto feature_offset = std::get<1>(index);
        auto fst = std::begin(replies[response_id].values()) + feature_offset;
        auto lst = fst + dimensions[feature_index];
        ++feature_index;
        if (feature_index == dimensions.size())
        {
            feature_index = 0;
        }

        out_values.insert(std::end(out_values), fst, lst);
    }
}

template <typename SparseFeaturesReply>
void UpdateSparseFeatureIndex(SparseFeaturesReply &reply, std::vector<snark::SparseFeatureIndex> &response_index,
                              size_t shard, std::span<int64_t> out_dimensions, size_t feature_count, std::mutex &mtx,
                              std::shared_ptr<Logger> logger = nullptr)
{
    if (reply.indices().empty())
    {
        return;
    }
    int64_t node_offset = 0;
    int64_t value_offset = 0;
    int64_t node_cummulative_count = 0;
    for (int64_t feature_index = 0; feature_index < reply.dimensions().size(); ++feature_index)
    {
        const auto feature_dim = reply.dimensions(feature_index) + 1;
        if (feature_dim == 1)
        {
            continue;
        }

        std::lock_guard lock(mtx);
        if (out_dimensions[feature_index] != 0 && reply.dimensions(feature_index) != 0 &&
            out_dimensions[feature_index] != reply.dimensions(feature_index))
        {
            auto feature_str = std::to_string(feature_index);
            auto client_dimension_str = std::to_string(out_dimensions[feature_index]);
            auto server_dimension_str = std::to_string(reply.dimensions(feature_index));
            logger->log_fatal("Dimensions do not match for sparse feature %s. %s != %s", feature_str.c_str(),
                              client_dimension_str.c_str(), server_dimension_str.c_str());
        }

        out_dimensions[feature_index] = reply.dimensions(feature_index);
        const size_t value_increment =
            (reply.values_counts(feature_index) * feature_dim) / reply.indices_counts(feature_index);
        node_cummulative_count += reply.indices_counts(feature_index);
        for (; node_offset < node_cummulative_count; node_offset += feature_dim, value_offset += value_increment)
        {
            const auto item_index = reply.indices(node_offset);
            auto &response_index_item = response_index[item_index * feature_count + feature_index];
            if (reply.timestamps(item_index * feature_count + feature_index) <= response_index_item.timestamp ||
                (response_index_item.shard >= 0 && response_index_item.shard != int(shard)))
            {
                continue;
            }

            if (response_index_item.index_count == 0)
            {
                response_index_item.shard = shard;
                response_index_item.index_offset = node_offset;
                response_index_item.value_offset = value_offset;
            }
            response_index_item.index_count += feature_dim;
            response_index_item.value_count += value_increment;
        }
    }
}

template <typename StringFeaturesReply>
void UpdateStringFeatureIndex(StringFeaturesReply &reply, std::mutex &mtx,
                              std::vector<std::tuple<size_t, size_t, snark::Timestamp>> &response_index, size_t shard,
                              std::span<int64_t> out_dimensions)
{
    if (reply.values().empty())
    {
        return;
    }

    int64_t value_offset = 0;
    int64_t ts_index = 0;
    for (int64_t feature_index = 0; feature_index < reply.dimensions().size(); ++feature_index)
    {
        const auto dim = reply.dimensions(feature_index);
        if (dim == 0)
        {
            continue;
        }

        std::lock_guard lock(mtx);
        const auto ts = reply.timestamps(ts_index);
        ++ts_index;
        if (std::get<2>(response_index[feature_index]) <= ts)
        {
            response_index[feature_index] = {shard, value_offset, ts};
            out_dimensions[feature_index] = dim;
        }

        value_offset += dim;
    }
}

template <typename LastNCreatedNeighborsReply>
void MergeLastNCreatedNeighbors(size_t count, std::vector<LastNCreatedNeighborsReply> &replies,
                                std::span<const NodeId> input_node_ids, std::atomic<size_t> &responses_left,
                                std::span<NodeId> output_neighbor_ids, std::span<Type> output_neighbor_types,
                                std::span<float> neighbors_weights, std::span<Timestamp> output_timestamps,
                                NodeId default_node_id, float default_weight, Type default_edge_type,
                                Timestamp default_timestamp, bool return_edge_created_ts)
{
    // Skip processing until all responses arrived. All responses are stored in the `replies` variable,
    // so we can safely return.
    if (responses_left.fetch_sub(1) > 1)
    {
        return;
    }

    for (size_t node_index = 0; node_index < input_node_ids.size(); ++node_index)
    {
        using ts_position = std::pair<Timestamp, size_t>;
        auto out_nodes = output_neighbor_ids.subspan(count * node_index, count);
        auto out_types = output_neighbor_types.subspan(count * node_index, count);
        auto out_weights = neighbors_weights.subspan(count * node_index, count);
        auto out_ts = output_timestamps; // to avoid trigger assert in a debug builds.
        if (return_edge_created_ts)
        {
            out_ts = output_timestamps.subspan(count * node_index, count);
        }
        std::priority_queue<ts_position, std::vector<ts_position>, std::greater<ts_position>> lastn;
        for (size_t reply_index = 0; reply_index < std::size(replies); ++reply_index)
        {
            const auto &reply = replies[reply_index];
            const auto nb_count = reply.neighbor_counts(node_index);
            if (nb_count == 0)
            {
                continue;
            }

            const size_t start_offset = node_index * count;
            for (size_t nb_index = start_offset; nb_index < start_offset + nb_count; ++nb_index)
            {
                auto ts = reply.timestamps(nb_index);
                size_t pos = lastn.size();
                if (lastn.size() == count)
                {
                    auto top = lastn.top();
                    if (top.first >= ts)
                    {
                        continue;
                    }
                    else
                    {
                        lastn.pop();
                    }

                    pos = top.second;
                }

                lastn.emplace(ts, pos);
                out_nodes[pos] = reply.node_ids(nb_index);
                out_types[pos] = reply.edge_types(nb_index);
                out_weights[pos] = reply.edge_weights(nb_index);
                if (return_edge_created_ts)
                {
                    out_ts[pos] = ts;
                }
            }
        }

        if (lastn.size() < count)
        {
            const auto start = lastn.size();
            std::fill(std::begin(out_nodes) + start, std::end(out_nodes), default_node_id);
            std::fill(std::begin(out_types) + start, std::end(out_types), default_edge_type);
            std::fill(std::begin(out_weights) + start, std::end(out_weights), default_weight);
            if (return_edge_created_ts)
            {
                std::fill(std::begin(out_ts) + start, std::end(out_ts), default_timestamp);
            }
        }
    }
};

template <typename WeightedSampleNeighborsReply>
void MergeWeightedSampledNeighbors(const WeightedSampleNeighborsReply &reply, std::mutex &mtx, size_t count,
                                   std::span<NodeId> output_neighbors, std::span<Type> output_types,
                                   std::span<float> output_weights, std::span<snark::Timestamp> output_edge_created_ts,
                                   std::span<const NodeId> node_ids, snark::Xoroshiro128PlusGenerator &engine,
                                   std::vector<float> &shard_weights, NodeId default_node_id, float default_weight,
                                   Type default_edge_type, bool return_edge_created_ts)
{
    if (reply.node_ids().empty())
    {
        return;
    }

    auto curr_nodes = std::begin(node_ids);
    auto curr_out_neighbor = std::begin(output_neighbors);
    auto curr_out_type = std::begin(output_types);
    auto curr_out_ts = std::begin(output_edge_created_ts);
    auto curr_out_weight = std::begin(output_weights);
    auto curr_shard_weight = std::begin(shard_weights);

    auto curr_reply_neighbor = std::begin(reply.neighbor_ids());
    auto curr_reply_type = std::begin(reply.neighbor_types());
    auto curr_reply_ts = std::begin(reply.timestamps());
    auto curr_reply_weight = std::begin(reply.neighbor_weights());
    auto curr_reply_shard_weight = std::begin(reply.shard_weights());
    boost::random::uniform_real_distribution<float> selector(0, 1);

    // We need to lock the merge in case some nodes are present in multiple
    // servers(a super node with lots of neighbors). To keep contention low
    // we'll use a global lock per response instead of per node.
    std::lock_guard lock(mtx);

    // The strategy is to zip nodes from server response matching to the input
    // nodes.
    for (const auto &reply_node_id : reply.node_ids())
    {
        // Loop until we find a match
        for (; curr_nodes != std::end(node_ids) && *curr_nodes != reply_node_id; ++curr_nodes)
        {
            curr_out_neighbor += count;
            curr_out_weight += count;
            curr_out_type += count;
            if (return_edge_created_ts)
            {
                curr_out_ts += count;
            }
            ++curr_shard_weight;
        }

        *curr_shard_weight += *curr_reply_shard_weight;
        if (*curr_shard_weight == 0)
        {
            ++curr_shard_weight;
            ++curr_reply_shard_weight;

            curr_out_neighbor = std::fill_n(curr_out_neighbor, count, default_node_id);
            curr_out_weight = std::fill_n(curr_out_weight, count, default_weight);
            curr_out_type = std::fill_n(curr_out_type, count, default_edge_type);
            if (return_edge_created_ts)
            {
                curr_out_ts = std::fill_n(curr_out_ts, count, PLACEHOLDER_TIMESTAMP);
                curr_reply_ts += count;
            }

            curr_reply_neighbor += count;
            curr_reply_type += count;
            curr_reply_weight += count;
            ++curr_nodes;
            continue;
        }

        float overwrite_rate = *curr_reply_shard_weight / *curr_shard_weight;
        for (size_t i = 0; i < count; ++i)
        {
            // Perf optimization for the most common scenario: every node has its
            // neighbors in one partition.
            if (overwrite_rate < 1.0f && selector(engine) > overwrite_rate)
            {
                ++curr_reply_weight;
                ++curr_reply_neighbor;
                ++curr_reply_type;
                ++curr_out_neighbor;
                ++curr_out_type;
                if (return_edge_created_ts)
                {
                    ++curr_out_ts;
                    ++curr_reply_ts;
                }
                ++curr_out_weight;
                continue;
            }

            *(curr_out_neighbor++) = *(curr_reply_neighbor++);
            *(curr_out_type++) = *(curr_reply_type++);
            *(curr_out_weight++) = *(curr_reply_weight++);
            if (return_edge_created_ts)
            {
                *(curr_out_ts++) = *(curr_reply_ts++);
            }
        }

        ++curr_reply_shard_weight;
        ++curr_shard_weight;
        ++curr_nodes;
    }

    assert(curr_reply_weight == std::end(reply.neighbor_weights()));
    assert(curr_reply_ts == std::end(reply.timestamps()));
    assert(curr_reply_neighbor == std::end(reply.neighbor_ids()));
    assert(curr_reply_type == std::end(reply.neighbor_types()));
    assert(curr_reply_shard_weight == std::end(reply.shard_weights()));
}

} // namespace snark

#endif // SNARK_MERGER_H
