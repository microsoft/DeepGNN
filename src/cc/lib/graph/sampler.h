// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#ifndef SNARK_SAMPLER_H
#define SNARK_SAMPLER_H

#include <cstdlib>
#include <memory>
#include <mutex>
#include <set>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"

#include "metadata.h"
#include "types.h"

namespace snark
{

// Helper function to adjust vector weights suitable for
// multinomial sampling from partitions.
void conditional_probabilities(std::vector<float> &w);

// https://en.wikipedia.org/wiki/Reservoir_sampling
// Complexity: O(k * (1 + log(N/k)))
// Another possible implementation Format-Preserving Encryption
// https://stackoverflow.com/a/16097246 it will be more complex though.
void SampleWithoutReplacement(int64_t seed, std::vector<std::span<const NodeId>> population,
                              std::vector<std::span<NodeId>> out, float overwrite_rate);

void SampleWithReplacement(int64_t seed, std::vector<std::span<const NodeId>> population,
                           std::vector<std::span<NodeId>> out, float overwrite_rate);

class Sampler
{
  public:
    virtual void Sample(int64_t seed, std::span<Type> out_types, std::span<NodeId> out_nodes, ...) const = 0;

    virtual float Weight() const = 0;

    virtual ~Sampler() = default;
};

class SamplerFactory
{
  public:
    virtual std::unique_ptr<Sampler> Create(std::set<Type> tp, std::set<size_t> partitions) = 0;
    virtual ~SamplerFactory() = default;
};

enum SamplerElement
{
    Node,
    Edge
};

template <typename Partition, SamplerElement element = Node> class SamplerImpl final : public Sampler
{
  public:
    SamplerImpl(SamplerImpl &&) = default;
    SamplerImpl(std::vector<Type> types, std::vector<std::shared_ptr<std::vector<Partition>>> partitions);

    void Sample(int64_t seed, std::span<Type> out_types, std::span<NodeId> out, ...) const override;

    float Weight() const override;

  private:
    // Types this sampler holds
    std::vector<Type> m_types;

    // Probability p[i] of selection of a type m_types[i].
    std::vector<float> m_type_weights;

    // Probability m_partition_weights[i][p] to select type m_types[i] and
    // partition p.
    std::vector<std::vector<float>> m_partition_weights;

    // Alias tables partitions with actual nodes.
    std::vector<std::shared_ptr<std::vector<Partition>>> m_partitions;

    // Total weight of elements in this sampler;
    float m_total_weight;

    // WithReplacement sampling
    // Total number of elements can be picked with type m_types[i].
    std::vector<size_t> m_type_min;
    std::vector<size_t> m_type_max;

    // Number of elements can be picked from m_partition_counts[i][p]
    // with type m_types[i] located in partition p.
    std::vector<std::vector<size_t>> m_partition_min;
    std::vector<std::vector<size_t>> m_partition_max;
};

template <typename Partition, SamplerElement element = Node> class AbstractSamplerFactory final : public SamplerFactory
{
  public:
    explicit AbstractSamplerFactory(std::string path);

    // We are using a regular set to keep type order consistent.
    std::unique_ptr<Sampler> Create(std::set<Type> tp, std::set<size_t> partitions) override;

  private:
    void Read(Type type, const std::set<size_t> &partitions);

    Metadata m_metadata;

    // Types loaded in the sampler. We store them in shared_ptrs because
    // we want to load a type only once for any possible types permutations.
    absl::flat_hash_map<Type, std::shared_ptr<std::vector<Partition>>> m_types;

    // Synchronize loading of partitions inside the factory.
    std::mutex m_mtx;
};

struct WeightedNodeSamplerRecord
{
    NodeId m_left;
    NodeId m_right;
    float m_threshold;
};

class WeightedNodeSamplerPartition
{
  public:
    WeightedNodeSamplerPartition(WeightedNodeSamplerPartition &&) = default;
    WeightedNodeSamplerPartition(const WeightedNodeSamplerPartition &) = default;
    WeightedNodeSamplerPartition(std::vector<WeightedNodeSamplerRecord> records, float weight);

    WeightedNodeSamplerPartition(Metadata meta, Type tp, size_t partition);

    void Sample(int64_t seed, std::span<NodeId> out) const;

    float Weight() const;
    bool Replacement() const;

  private:
    std::vector<WeightedNodeSamplerRecord> m_records;
    float m_weight;
};

struct WeightedEdgeSamplerRecord
{
    NodeId m_left_src;
    NodeId m_left_dst;
    NodeId m_right_src;
    NodeId m_right_dst;
    float m_threshold;
};

class WeightedEdgeSamplerPartition
{
  public:
    WeightedEdgeSamplerPartition(WeightedEdgeSamplerPartition &&) = default;
    WeightedEdgeSamplerPartition(const WeightedEdgeSamplerPartition &) = default;
    WeightedEdgeSamplerPartition(std::vector<WeightedEdgeSamplerRecord> records, float weight);

    WeightedEdgeSamplerPartition(Metadata meta, Type tp, size_t partition);

    void Sample(int64_t seed, std::span<NodeId> out_src, std::span<NodeId> out_dst) const;
    float Weight() const;
    bool Replacement() const;

  private:
    std::vector<WeightedEdgeSamplerRecord> m_records;
    float m_weight;
};

template <bool WithReplacement> class UniformEdgeSamplerPartition
{
  public:
    UniformEdgeSamplerPartition(UniformEdgeSamplerPartition &&) = default;
    UniformEdgeSamplerPartition(const UniformEdgeSamplerPartition &) = default;
    explicit UniformEdgeSamplerPartition(std::vector<std::pair<NodeId, NodeId>> records);
    UniformEdgeSamplerPartition(Metadata meta, Type tp, size_t partition);

    void Sample(int64_t seed, std::span<NodeId> out_src, std::span<NodeId> out_dst) const;
    float Weight() const;
    bool Replacement() const;

  private:
    std::vector<NodeId> m_src;
    std::vector<NodeId> m_dst;
    float m_weight;
};

template <bool WithReplacement> class UniformNodeSamplerPartition
{
  public:
    UniformNodeSamplerPartition(UniformNodeSamplerPartition &&) = default;
    UniformNodeSamplerPartition(const UniformNodeSamplerPartition &) = default;
    explicit UniformNodeSamplerPartition(std::vector<NodeId> records);
    UniformNodeSamplerPartition(Metadata meta, Type tp, size_t partition);

    void Sample(int64_t seed, std::span<NodeId> out) const;
    float Weight() const;
    bool Replacement() const;

  private:
    std::vector<NodeId> m_records;
    float m_weight;
};

using WeightedEdgeSampler = SamplerImpl<WeightedEdgeSamplerPartition, Edge>;
using WeightedEdgeSamplerFactory = AbstractSamplerFactory<WeightedEdgeSamplerPartition, Edge>;

using UniformEdgeSamplerFactoryWithoutReplacement = AbstractSamplerFactory<UniformEdgeSamplerPartition<false>, Edge>;
using UniformEdgeSamplerWithoutReplacement = SamplerImpl<UniformEdgeSamplerPartition<false>, Edge>;

using UniformEdgeSamplerFactory = AbstractSamplerFactory<UniformEdgeSamplerPartition<true>, Edge>;
using UniformEdgeSampler = SamplerImpl<UniformEdgeSamplerPartition<true>, Edge>;

using WeightedNodeSamplerFactory = AbstractSamplerFactory<WeightedNodeSamplerPartition>;
using WeightedNodeSampler = SamplerImpl<WeightedNodeSamplerPartition>;

using UniformNodeSamplerFactoryWithoutReplacement = AbstractSamplerFactory<UniformNodeSamplerPartition<false>>;
using UniformNodeSamplerWithoutReplacement = SamplerImpl<UniformNodeSamplerPartition<false>>;

using UniformNodeSamplerFactory = AbstractSamplerFactory<UniformNodeSamplerPartition<true>>;
using UniformNodeSampler = SamplerImpl<UniformNodeSamplerPartition<true>>;

} // namespace snark
#endif
