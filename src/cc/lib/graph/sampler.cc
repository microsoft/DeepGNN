// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "sampler.h"
#include <algorithm>
#include <cassert>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <iterator>
#include <numeric>
#include <random>
#include <type_traits>

#include "locator.h"
#include "storage.h"
#include "xoroshiro.h"

#include "absl/container/flat_hash_set.h"
#include "boost/random/binomial_distribution.hpp"
#include "boost/random/uniform_int_distribution.hpp"
#include "boost/random/uniform_real_distribution.hpp"
#include <glog/logging.h>
#include <glog/raw_logging.h>

namespace snark
{

template <typename Partition, SamplerElement element>
SamplerImpl<Partition, element>::SamplerImpl(std::vector<snark::Type> types,
                                             std::vector<std::shared_ptr<std::vector<Partition>>> partitions)
    : m_types(std::move(types)), m_partitions(std::move(partitions)), m_total_weight(0)
{
    for (auto &tp : m_partitions)
    {
        float total_weight = 0;
        m_partition_weights.emplace_back();
        m_partition_max.emplace_back();
        m_partition_min.emplace_back();
        for (auto &p : *tp)
        {
            const auto weight = p.Weight();
            m_partition_weights.back().emplace_back(weight);
            m_partition_max.back().emplace_back(weight);
            total_weight += weight;
        }

        conditional_probabilities(m_partition_weights.back());

        m_partition_min.back().resize(m_partition_max.back().size());
        auto last = m_partition_max.back().rbegin();

        // Aggregate max number of elements can be sampled from the remaining partitions.
        // E.g. if there are 3 partitions with 5, 6 and 7 elements, then
        // m_partition_max should be [13, 7, 0]
        std::partial_sum(last, last + m_partition_max.back().size() - 1, m_partition_min.back().rbegin() + 1);

        m_type_weights.emplace_back(total_weight);
        m_type_max.emplace_back(total_weight);
        m_total_weight += total_weight;
    }

    conditional_probabilities(m_type_weights);
    m_type_min.resize(m_type_max.size());
    auto last = m_type_max.rbegin();

    // Similar calculation as for partiotion, but for types.
    std::partial_sum(last, last + m_type_max.size() - 1, m_type_min.rbegin() + 1);
}

// First find out how many element of a specific type to sample with
// multinomial distribution. Use the same logic but for partitions. There are
// no standard multinomial distributions in C++, so we are using a sequence of
// binomial ones.
template <typename Partition, SamplerElement element>
void SamplerImpl<Partition, element>::Sample(int64_t seed, std::span<Type> out_types, std::span<NodeId> out, ...) const
{
    if (m_partitions.empty())
    {
        return;
    }

    snark::Xoroshiro128PlusGenerator gen(seed);

    // If we simply pass seed to subpartitions results will be correlated.
    // So we are going to create a separate seed generator.
    boost::random::uniform_int_distribution<int64_t> subseed;
    size_t left = out_types.size();
    auto curr_type = std::begin(out_types);
    size_t position = 0; // current position in output spans

    for (size_t i = 0; i < m_types.size() && left > 0; ++i)
    {
        size_t type_count = boost::random::binomial_distribution<int32_t>(left, m_type_weights[i])(gen);
        if (!m_partitions.front()->front().Replacement())
        {
            // We can't subsample more elements than a type has.
            type_count = std::min(type_count, m_type_max[i]);

            // Make sure we don't sample too little.
            if (left - type_count > m_type_min[i])
            {
                type_count = m_type_max[i];
            }
        }

        left -= type_count;
        curr_type = std::fill_n(curr_type, type_count, m_types[i]);
        size_t left_items = type_count;
        for (size_t p = 0; left_items > 0 && m_partitions[i] && p < m_partitions[i]->size(); ++p)
        {
            size_t item_count =
                boost::random::binomial_distribution<int32_t>(left_items, m_partition_weights[i][p])(gen);

            if (!m_partitions.front()->front().Replacement())
            {
                // We can't subsample more elements than a partition has.
                item_count = std::min(item_count, m_partition_max[i][p]);

                // Make sure we don't sample too little and handle larger than sample size output buffer.
                if (left_items - item_count > m_partition_min[i][p])
                {
                    item_count = m_partition_max[i][p];
                }
            }

            if (item_count == 0)
            {
                continue;
            }
            left_items -= item_count;
            if constexpr (element == SamplerElement::Node)
            {
                (*m_partitions[i])[p].Sample(subseed(gen), out.subspan(position, item_count));
            }
            else if constexpr (element == SamplerElement::Edge)
            {
                va_list args;
                va_start(args, out);
                auto out_dst = va_arg(args, std::span<NodeId>);
                (*m_partitions[i])[p].Sample(subseed(gen), out.subspan(position, item_count),
                                             out_dst.subspan(position, item_count));
                va_end(args);
            }

            position += item_count;
        }
    }
}

template <typename Partition, SamplerElement element> float SamplerImpl<Partition, element>::Weight() const
{
    return m_total_weight;
}

template <typename Partition, SamplerElement element>
AbstractSamplerFactory<Partition, element>::AbstractSamplerFactory(std::string path) : m_metadata(path)
{
}

template <typename Partition, SamplerElement element>
std::unique_ptr<Sampler> AbstractSamplerFactory<Partition, element>::Create(std::set<Type> tp,
                                                                            std::set<size_t> partition_indices)
{
    std::vector<Type> types;
    std::vector<std::shared_ptr<std::vector<Partition>>> partitions;
    {
        std::lock_guard guard(m_mtx);

        for (auto t : tp)
        {
            bool correct_type = t >= 0;
            if constexpr (element == SamplerElement::Node)
            {
                correct_type = correct_type && size_t(t) < m_metadata.m_node_type_count;
            }
            else
            {
                correct_type = correct_type && size_t(t) < m_metadata.m_edge_type_count;
            }

            if (!correct_type)
            {
                RAW_LOG_ERROR("Requested an unknown type %d", t);
                continue;
            }

            if (!m_types.contains(t))
            {
                Read(t, partition_indices);
            }

            types.emplace_back(t);
            partitions.emplace_back(m_types[t]);
        }
    }

    return std::make_unique<SamplerImpl<Partition, element>>(std::move(types), std::move(partitions));
}

template <typename Partition, SamplerElement element>
void AbstractSamplerFactory<Partition, element>::Read(Type type, const std::set<size_t> &partition_indices)
{
    std::vector<Partition> res;
    for (auto p : partition_indices)
    {
        res.emplace_back(m_metadata, type, p);
    }

    m_types[type] = std::make_shared<std::vector<Partition>>(std::move(res));
}

WeightedNodeSamplerPartition::WeightedNodeSamplerPartition(Metadata meta, Type tp, size_t partition)
    : m_weight(meta.m_partition_node_weights[partition][tp])
{
    std::shared_ptr<BaseStorage<uint8_t>> node_weights;
    if (!is_hdfs_path(meta.m_path))
    {
        node_weights = std::make_shared<DiskStorage<uint8_t>>(meta.m_path, partition, tp, open_node_alias);
    }
    else
    {
        auto full_path = std::filesystem::path(meta.m_path) /
                         ("node_" + std::to_string(tp) + "_" + std::to_string(partition) + ".alias");
        node_weights = std::make_shared<HDFSStreamStorage<uint8_t>>(full_path.c_str(), meta.m_config_path);
    }
    auto node_weights_ptr = node_weights->start();
    size_t record_size = node_weights->size() / (2 * sizeof(NodeId) + sizeof(float));

    m_records.reserve(record_size);
    for (size_t i = 0; i < record_size; ++i)
    {
        m_records.emplace_back();
        auto &back = m_records.back();
        if (1 != node_weights->read(&back.m_left, 8, 1, node_weights_ptr))
        {
            RAW_LOG_FATAL("Failed to read node from alias table");
        }
        if (1 != node_weights->read(&back.m_right, 8, 1, node_weights_ptr))
        {
            RAW_LOG_FATAL("Failed to read alias from alias table");
        }
        if (1 != node_weights->read(&back.m_threshold, 4, 1, node_weights_ptr))
        {
            RAW_LOG_FATAL("Failed to read probability from alias table");
        }
    }
}

WeightedNodeSamplerPartition::WeightedNodeSamplerPartition(std::vector<WeightedNodeSamplerRecord> records, float weight)
    : m_records(std::move(records)), m_weight{weight}
{
}

void WeightedNodeSamplerPartition::Sample(int64_t seed, std::span<NodeId> out) const
{
    if (m_records.empty())
    {
        return;
    }

    snark::Xoroshiro128PlusGenerator gen(seed);
    boost::random::uniform_real_distribution<float> toss(0, 1.0f);
    for (auto &n : out)
    {
        // Use toss mutliple times instead of uniform_int_distribution for
        // performance reasons.
        const auto &record = m_records[toss(gen) * m_records.size()];
        n = toss(gen) < record.m_threshold ? record.m_left : record.m_right;
    }
}

float WeightedNodeSamplerPartition::Weight() const
{
    return m_weight;
}

bool WeightedNodeSamplerPartition::Replacement() const
{
    return true;
}

template <bool WithReplacement>
UniformNodeSamplerPartition<WithReplacement>::UniformNodeSamplerPartition(Metadata meta, Type tp, size_t partition)
{
    absl::flat_hash_set<NodeId> node_set;
    std::shared_ptr<BaseStorage<uint8_t>> node_weights;
    if (!is_hdfs_path(meta.m_path))
    {
        node_weights = std::make_shared<DiskStorage<uint8_t>>(meta.m_path, partition, tp, open_node_alias);
    }
    else
    {
        auto full_path = std::filesystem::path(meta.m_path) /
                         ("node_" + std::to_string(tp) + "_" + std::to_string(partition) + ".alias");
        node_weights = std::make_shared<HDFSStreamStorage<uint8_t>>(full_path.c_str(), meta.m_config_path);
    }
    auto node_weights_ptr = node_weights->start();
    size_t record_size = node_weights->size() / (2 * sizeof(NodeId) + sizeof(float));

    m_records.reserve(record_size);
    for (size_t i = 0; i < record_size; ++i)
    {
        NodeId node;
        float prob;
        if (1 != node_weights->read(&node, sizeof(NodeId), 1, node_weights_ptr))
        {
            RAW_LOG_FATAL("Failed to read node from alias file");
        }
        node_set.insert(node);
        if (1 != node_weights->read(&node, sizeof(NodeId), 1, node_weights_ptr))
        {
            RAW_LOG_FATAL("Failed to read node from alias file");
        }
        if (1 != node_weights->read(&prob, sizeof(float), 1, node_weights_ptr))
        {
            RAW_LOG_FATAL("Failed to read probability from alias table");
        }
        // check for dummy node
        if (prob < 1.0)
        {
            node_set.insert(node);
        }
    }
    m_records.reserve(node_set.size());
    for (const auto &node : node_set)
    {
        m_records.emplace_back(node);
    }
    m_weight = m_records.size();
}

WeightedEdgeSamplerPartition::WeightedEdgeSamplerPartition(std::vector<WeightedEdgeSamplerRecord> records, float weight)
    : m_records(std::move(records)), m_weight{weight}
{
}

WeightedEdgeSamplerPartition::WeightedEdgeSamplerPartition(Metadata meta, Type tp, size_t partition)
    : m_weight(meta.m_partition_edge_weights[partition][tp])
{
    std::shared_ptr<BaseStorage<uint8_t>> edge_weights;
    if (!is_hdfs_path(meta.m_path))
    {
        edge_weights = std::make_shared<DiskStorage<uint8_t>>(meta.m_path, partition, tp, open_edge_alias);
    }
    else
    {
        auto full_path = std::filesystem::path(meta.m_path) /
                         ("edge_" + std::to_string(tp) + "_" + std::to_string(partition) + ".alias");
        edge_weights = std::make_shared<HDFSStreamStorage<uint8_t>>(full_path.c_str(), meta.m_config_path);
    }
    auto edge_weights_ptr = edge_weights->start();
    size_t record_size = edge_weights->size() / (4 * sizeof(NodeId) + sizeof(float));

    m_records.reserve(record_size);
    for (size_t i = 0; i < record_size; ++i)
    {
        NodeId record[4];
        if (4 != edge_weights->read(&record, sizeof(NodeId), 4, edge_weights_ptr))
        {
            RAW_LOG_FATAL("Failed to read record from alias file");
        }

        float threshold;
        if (1 != edge_weights->read(&threshold, sizeof(float), 1, edge_weights_ptr))
        {
            RAW_LOG_FATAL("Failed to read threshold from edge alias table");
        }

        m_records.emplace_back(WeightedEdgeSamplerRecord{record[0], record[1], record[2], record[3], threshold});
    }
}

void WeightedEdgeSamplerPartition::Sample(int64_t seed, std::span<NodeId> out_src, std::span<NodeId> out_dst) const
{
    if (m_records.empty())
    {
        return;
    }

    snark::Xoroshiro128PlusGenerator gen(seed);
    boost::random::uniform_real_distribution<float> toss(0, 1.0f);
    for (size_t i = 0; i < out_src.size(); ++i)
    {
        size_t pick = toss(gen) * m_records.size();
        auto &record = m_records[pick];
        if (toss(gen) < record.m_threshold)
        {
            out_src[i] = record.m_left_src;
            out_dst[i] = record.m_left_dst;
        }
        else
        {
            out_src[i] = record.m_right_src;
            out_dst[i] = record.m_right_dst;
        }
    }
}

float WeightedEdgeSamplerPartition::Weight() const
{
    return m_weight;
}

bool WeightedEdgeSamplerPartition::Replacement() const
{
    return true;
}

template <bool WithReplacement>
UniformNodeSamplerPartition<WithReplacement>::UniformNodeSamplerPartition(std::vector<NodeId> records)
    : m_records(std::move(records)), m_weight{float(m_records.size())}
{
}

template <bool WithReplacement> float UniformNodeSamplerPartition<WithReplacement>::Weight() const
{
    return m_weight;
}

template <bool WithReplacement> bool UniformNodeSamplerPartition<WithReplacement>::Replacement() const
{
    return WithReplacement;
}

template <bool WithReplacement>
void UniformNodeSamplerPartition<WithReplacement>::Sample(int64_t seed, std::span<NodeId> out) const
{
    if (m_records.empty())
    {
        return;
    }

    if constexpr (WithReplacement)
    {
        SampleWithReplacement(seed, {std::span(m_records)}, {out}, 1.f);
    }
    else
    {
        SampleWithoutReplacement(seed, {std::span(m_records)}, {out}, 1.f);
    }
}

template <bool WithReplacement>
UniformEdgeSamplerPartition<WithReplacement>::UniformEdgeSamplerPartition(Metadata meta, Type tp, size_t partition)
{
    struct pair_hash
    {
        size_t operator()(const std::pair<int, int> &p) const
        {
            return 48271 * p.first + p.second;
        }
    };
    absl::flat_hash_set<std::pair<NodeId, NodeId>, pair_hash> edge_set;
    std::shared_ptr<BaseStorage<uint8_t>> edge_alias;
    if (!is_hdfs_path(meta.m_path))
    {
        edge_alias = std::make_shared<DiskStorage<uint8_t>>(meta.m_path, partition, tp, open_edge_alias);
    }
    else
    {
        auto full_path = std::filesystem::path(meta.m_path) /
                         ("edge_" + std::to_string(tp) + "_" + std::to_string(partition) + ".alias");
        edge_alias = std::make_shared<HDFSStreamStorage<uint8_t>>(full_path.c_str(), meta.m_config_path);
    }

    auto edge_alias_ptr = edge_alias->start();
    size_t record_size = edge_alias->size() / (4 * sizeof(NodeId) + sizeof(float));
    for (size_t i = 0; i < record_size; ++i)
    {
        NodeId src, dst;
        float prob;
        if (1 != edge_alias->read(&src, sizeof(NodeId), 1, edge_alias_ptr))
        {
            RAW_LOG_FATAL("Failed to read left edge source from alias file");
        }
        if (1 != edge_alias->read(&dst, sizeof(NodeId), 1, edge_alias_ptr))
        {
            RAW_LOG_FATAL("Failed to read left edge destination from alias file");
        }
        edge_set.insert(std::make_pair(src, dst));

        if (1 != edge_alias->read(&src, sizeof(NodeId), 1, edge_alias_ptr))
        {
            RAW_LOG_FATAL("Failed to read right edge source from alias file");
        }
        if (1 != edge_alias->read(&dst, sizeof(NodeId), 1, edge_alias_ptr))
        {
            RAW_LOG_FATAL("Failed to read right edge destination from alias file");
        }
        if (1 != edge_alias->read(&prob, sizeof(float), 1, edge_alias_ptr))
        {
            RAW_LOG_FATAL("Failed to read probability from edge alias file");
        }

        // check for dummy edge
        if (prob < 1.0)
        {
            edge_set.insert(std::make_pair(src, dst));
        }
    }

    m_src.reserve(edge_set.size());
    m_dst.reserve(edge_set.size());
    for (const auto &edge : edge_set)
    {
        m_src.emplace_back(edge.first);
        m_dst.emplace_back(edge.second);
    }
    m_weight = float(edge_set.size());
}

template <bool WithReplacement>
UniformEdgeSamplerPartition<WithReplacement>::UniformEdgeSamplerPartition(
    std::vector<std::pair<NodeId, NodeId>> records)
    : m_weight{float(records.size())}
{
    for (auto n : records)
    {
        m_src.emplace_back(n.first);
        m_dst.emplace_back(n.second);
    }
}

template <bool WithReplacement> float UniformEdgeSamplerPartition<WithReplacement>::Weight() const
{
    return m_weight;
}

template <bool WithReplacement> bool UniformEdgeSamplerPartition<WithReplacement>::Replacement() const
{
    return WithReplacement;
}

template <bool WithReplacement>
void UniformEdgeSamplerPartition<WithReplacement>::Sample(int64_t seed, std::span<NodeId> out_src,
                                                          std::span<NodeId> out_dst) const
{
    if (m_src.empty() || m_dst.empty())
    {
        return;
    }

    if constexpr (WithReplacement)
    {
        SampleWithReplacement(seed, {std::span(m_src), std::span(m_dst)}, {out_src, out_dst}, 1.f);
    }
    else
    {
        SampleWithoutReplacement(seed, {std::span(m_src), std::span(m_dst)}, {out_src, out_dst}, 1.f);
    }
}

void SampleWithoutReplacement(int64_t seed, std::vector<std::span<const NodeId>> population,
                              std::vector<std::span<NodeId>> out, float overwrite_rate)
{
    assert(!population.empty());
    assert(population.size() == out.size());
    assert(!population.front().empty());

    auto count = out.front().size();
    const auto population_size = population.front().size();

    snark::Xoroshiro128PlusGenerator gen(seed);
    boost::random::uniform_real_distribution<float> toss(0, 1.f);
    while (count >= population_size)
    {
        for (size_t index = 0; index < population.size(); ++index)
        {
            std::copy(std::begin(population[index]), std::end(population[index]), std::begin(out[index]));
        }
        for (size_t pos = 0; pos < population_size; ++pos)
        {
            if (overwrite_rate < 1.0f || toss(gen) > overwrite_rate)
            {
                continue;
            }
            for (size_t index = 0; index < population.size(); ++index)
            {
                out[index][pos] = population[index][pos];
            }
        }

        count -= population_size;
        for (size_t index = 0; index < population.size(); ++index)
        {
            out[index] = out[index].subspan(population_size);
        }
    }
    if (count == 0)
    {
        return;
    }

    for (size_t pos = 0; pos < std::min(count, population_size);)
    {
        if (overwrite_rate < 1.0f && toss(gen) > overwrite_rate)
        {
            for (size_t index = 0; index < population.size(); ++index)
            {
                std::swap(out[index][pos], out[index][count - 1]);
            }
            --count;
            continue;
        }
        for (size_t index = 0; index < population.size(); ++index)
        {
            out[index][pos] = population[index][pos];
        }
        ++pos;
    }
    if (count == 0)
    {
        return;
    }

    float w = std::exp(std::log(toss(gen)) / count);
    size_t i = count - 1;
    while (i < population_size)
    {
        i += std::floor(std::log(toss(gen)) / std::log(1 - w)) + 1;
        if (i < population_size)
        {
            const size_t pick = toss(gen) * count;
            for (size_t index = 0; index < population.size(); ++index)
            {
                out[index][pick] = population[index][i];
            }
            w = w * std::exp(std::log(toss(gen)) / count);
        }
    }
}

void SampleWithReplacement(int64_t seed, std::vector<std::span<const NodeId>> population,
                           std::vector<std::span<NodeId>> out, float overwrite_rate)
{
    assert(!population.empty());
    assert(!population.front().empty());
    assert(population.size() == out.size());

    snark::Xoroshiro128PlusGenerator gen(seed);
    boost::random::uniform_real_distribution<float> toss(0, 1);

    for (size_t pos = 0; pos < out.front().size(); ++pos)
    {
        if (overwrite_rate < 1.0f && toss(gen) > overwrite_rate)
        {
            continue;
        }
        const size_t pick = toss(gen) * population.front().size();
        for (size_t index = 0; index < out.size(); ++index)
        {
            out[index][pos] = population[index][pick];
        }
    }
}

void conditional_probabilities(std::vector<float> &probs)
{
    // Conditional probabilities for sequential sampling: after we are done with
    // type i probability for sampling of type i should be weight_{i+1} /
    // sum_{i+1}^{n}(weight_x), because we are not going to consider weights
    // <= i anymore. Final probabilities should be:
    // |    type_1      | type_2 |                 ... | type_n |
    // | w_1 / sum(w_i) | w_2 / (sum(w_i) - w_1) | ... |   1    |
    if (probs.empty())
    {
        return;
    }

    // Adjust probabilities for binomial sampling.
    float sum_weight = 0;
    for (auto p = std::rbegin(probs); p != std::rend(probs); p++)
    {
        sum_weight += *p;
        *p /= sum_weight;
    }
}

template class SamplerImpl<WeightedNodeSamplerPartition>;
template class AbstractSamplerFactory<WeightedNodeSamplerPartition>;

template class UniformNodeSamplerPartition<false>;
template class UniformNodeSamplerPartition<true>;
template class SamplerImpl<UniformNodeSamplerPartition<false>>;
template class AbstractSamplerFactory<UniformNodeSamplerPartition<false>>;

template class SamplerImpl<UniformNodeSamplerPartition<true>>;
template class AbstractSamplerFactory<UniformNodeSamplerPartition<true>>;

template class SamplerImpl<WeightedEdgeSamplerPartition, Edge>;
template class AbstractSamplerFactory<WeightedEdgeSamplerPartition, Edge>;

template class UniformEdgeSamplerPartition<false>;
template class UniformEdgeSamplerPartition<true>;
template class SamplerImpl<UniformEdgeSamplerPartition<false>, Edge>;
template class AbstractSamplerFactory<UniformEdgeSamplerPartition<false>, Edge>;

template class SamplerImpl<UniformEdgeSamplerPartition<true>, Edge>;
template class AbstractSamplerFactory<UniformEdgeSamplerPartition<true>, Edge>;
} // namespace snark
