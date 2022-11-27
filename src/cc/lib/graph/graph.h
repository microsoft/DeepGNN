// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#ifndef SNARK_GRAPH_H
#define SNARK_GRAPH_H

#include <cstdlib>
#include <memory>
#include <random>
#include <span>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "partition.h"
#include "sampler.h"
#include "types.h"

namespace snark
{

class Graph
{
  public:
    Graph(snark::Metadata metadata, std::vector<std::string> paths, std::vector<uint32_t> partitions,
          PartitionStorageType storage_type);

    void GetNodeType(std::span<const NodeId> node_ids, std::span<Type> output, Type default_type) const;

    void GetNodeFeature(std::span<const NodeId> node_ids, std::span<snark::FeatureMeta> features,
                        std::span<uint8_t> output) const;

    void GetNodeSparseFeature(std::span<const NodeId> node_ids, std::span<const snark::FeatureId> features,
                              std::span<int64_t> out_dimensions, std::vector<std::vector<int64_t>> &out_indices,
                              std::vector<std::vector<uint8_t>> &out_data) const;

    void GetNodeStringFeature(std::span<const NodeId> node_ids, std::span<const snark::FeatureId> features,
                              std::span<int64_t> out_dimensions, std::vector<uint8_t> &out_data) const;

    void GetEdgeFeature(std::span<const NodeId> input_edge_src, std::span<const NodeId> input_edge_dst,
                        std::span<const Type> input_edge_type, std::span<snark::FeatureMeta> features,
                        std::span<uint8_t> output) const;

    void GetEdgeSparseFeature(std::span<const NodeId> input_edge_src, std::span<const NodeId> input_edge_dst,
                              std::span<const Type> input_edge_type, std::span<const snark::FeatureId> features,
                              std::span<int64_t> out_dimensions, std::vector<std::vector<int64_t>> &out_indices,
                              std::vector<std::vector<uint8_t>> &out_values) const;

    void GetEdgeStringFeature(std::span<const NodeId> input_edge_src, std::span<const NodeId> input_edge_dst,
                              std::span<const Type> input_edge_type, std::span<const snark::FeatureId> features,
                              std::span<int64_t> out_dimensions, std::vector<uint8_t> &out_values) const;

    void NeighborCount(std::span<const NodeId> input_node_ids, std::span<const Type> input_edge_types,
                       std::span<uint64_t> output_neighbors_counts) const;

    void FullNeighbor(std::span<const NodeId> input_node_ids, std::span<const Type> input_edge_types,
                      std::vector<NodeId> &output_neighbor_ids, std::vector<Type> &output_neighbor_types,
                      std::vector<float> &output_neighbors_weights, std::span<uint64_t> output_neighbors_counts) const;

    void SampleNeighbor(int64_t seed, std::span<const NodeId> input_node_ids, std::span<Type> input_edge_types,
                        size_t count, std::span<NodeId> output_neighbor_ids, std::span<Type> output_neighbor_types,
                        std::span<float> neighbors_weights, std::span<float> neighbors_total_weights,
                        NodeId default_node_id, float default_weight, Type default_edge_type) const;

    void UniformSampleNeighbor(bool without_replacement, int64_t seed, std::span<const NodeId> input_node_ids,
                               std::span<Type> input_edge_types, size_t count, std::span<NodeId> output_neighbor_ids,
                               std::span<Type> output_neighbor_types, std::span<uint64_t> neighbors_total_count,
                               NodeId default_node_id, Type default_edge_type) const;

    Metadata GetMetadata() const;

  private:
    void ReadNodeMap(std::filesystem::path path, std::string suffix, uint32_t index);

    std::vector<std::tuple<std::size_t, std::size_t>> SplitIntoGroups(
        std::size_t count, std::size_t parts = std::thread::hardware_concurrency() - 1) const;
    bool UseThreadPoolWhenGettingFeatures(std::size_t count, std::size_t feature_size) const;
    bool UseThreadPoolWhenGettingNeighbors(std::size_t count, std::size_t neighbor_count) const;

    std::vector<std::shared_ptr<Partition>> m_partitions;
    absl::flat_hash_map<NodeId, uint64_t> m_node_map;
    std::vector<uint32_t> m_partitions_indices;
    std::vector<uint64_t> m_internal_indices;
    std::vector<uint32_t> m_counts;
    Metadata m_metadata;
};

} // namespace snark

#endif // SNARK_GRAPH_H
