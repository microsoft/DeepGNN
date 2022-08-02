// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#ifndef SNARK_PARTITION_H
#define SNARK_PARTITION_H
#include <cstdlib>
#include <filesystem>
#include <random>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "metadata.h"
#include "storage.h"
#include "types.h"
#include "xoroshiro.h"

#include "boost/random/uniform_real_distribution.hpp"

namespace snark
{
struct Partition
{
    Partition() = default;
    Partition(std::filesystem::path path, std::string suffix, PartitionStorageType storage_type);

    Type GetNodeType(uint64_t internal_node_id) const;
    void GetNodeFeature(uint64_t internal_node_id, std::span<snark::FeatureMeta> features,
                        std::span<uint8_t> output) const;
    void GetNodeSparseFeature(uint64_t internal_node_id, std::span<const snark::FeatureId> features, int64_t prefix,
                              std::span<int64_t> out_dimensions, std::vector<std::vector<int64_t>> &out_indices,
                              std::vector<std::vector<uint8_t>> &out_values) const;

    void GetNodeStringFeature(uint64_t internal_node_id, std::span<const snark::FeatureId> features,
                              std::span<int64_t> out_dimensions, std::vector<uint8_t> &out_values) const;

    // Return true if an edge was found in the partition
    bool GetEdgeFeature(uint64_t internal_src_node_id, NodeId input_edge_dst, Type input_edge_type,
                        std::span<snark::FeatureMeta> features, std::span<uint8_t> output) const;

    bool GetEdgeSparseFeature(uint64_t internal_src_node_id, NodeId input_edge_dst, Type input_edge_type,
                              std::span<const snark::FeatureId> features, int64_t prefix,
                              std::span<int64_t> out_dimensions, std::vector<std::vector<int64_t>> &out_indices,
                              std::vector<std::vector<uint8_t>> &out_values) const;

    bool GetEdgeStringFeature(uint64_t internal_src_node_id, NodeId input_edge_dst, Type input_edge_type,
                              std::span<const snark::FeatureId> features, std::span<int64_t> out_dimensions,
                              std::vector<uint8_t> &out_values) const;

    // Retrieve total number of neighbors with specified edge types and returns the total number
    // of such neighbors.
    size_t NeighborCount(uint64_t internal_node_id, std::span<const Type> edge_types) const;

    // Backfill out_* vectors with information about neighbors of the node
    // with id equal to node_id and returns total number of such neighbors.
    size_t FullNeighbor(uint64_t internal_node_id, std::span<const Type> edge_types,
                        std::vector<NodeId> &out_neighbors_ids, std::vector<Type> &out_edge_types,
                        std::vector<float> &out_edge_weights) const;

    // in_edge_types has to have types in strictly increasing order.
    // out_partition contains information about neighbor weights for a
    // particular node in that partition. This is useful in case node neighbors
    // are distributed accross multiple partitions.
    void SampleNeighbor(int64_t seed, uint64_t internal_node_id, std::span<const Type> in_edge_types, uint64_t count,
                        std::span<NodeId> out_nodes, std::span<Type> out_types, std::span<float> out_weights,
                        float &out_partition, NodeId default_node_id, float default_weight, Type default_type) const;

    // in_edge_types has to have types in strictly increasing order.
    void UniformSampleNeighbor(bool without_replacement, int64_t seed, uint64_t internal_node_id,
                               std::span<const Type> in_edge_types, uint64_t count, std::span<NodeId> out_nodes,
                               std::span<Type> out_types, uint64_t &out_partition_count, NodeId default_node_id,
                               Type default_edge_type) const;

    Metadata GetMetadata() const;

  private:
    void ReadNodeMap(std::filesystem::path path, std::string suffix);
    void ReadNodeIndex(std::filesystem::path path, std::string suffix);
    void ReadEdges(std::filesystem::path path, std::string suffix);
    void ReadNeighborsIndex(std::filesystem::path path, std::string suffix);
    void ReadEdgeIndex(std::filesystem::path path, std::string suffix);
    void ReadNodeFeatures(std::filesystem::path path, std::string suffix);
    void ReadNodeFeaturesIndex(std::filesystem::path path, std::string suffix);
    void ReadNodeFeaturesData(std::filesystem::path path, std::string suffix);
    void ReadEdgeFeaturesIndex(std::filesystem::path path, std::string suffix);
    void ReadEdgeFeaturesData(std::filesystem::path path, std::string suffix);

    void UniformSampleNeighborWithoutReplacement(int64_t seed, uint64_t internal_node_ids,
                                                 std::span<const Type> in_edge_types, uint64_t count,
                                                 std::span<NodeId> out_nodes, std::span<Type> out_types,
                                                 uint64_t &out_partition_count, NodeId default_node_id,
                                                 Type default_edge_type) const;
    void UniformSampleNeighborWithReplacement(int64_t seed, uint64_t internal_node_ids,
                                              std::span<const Type> in_edge_types, uint64_t count,
                                              std::span<NodeId> out_nodes, std::span<Type> out_types,
                                              uint64_t &out_partition_count, NodeId default_node_id,
                                              Type default_edge_type) const;
    void UniformSampleMergeWithoutReplacement(
        uint64_t count, std::vector<NodeId> &left_neighbors, std::vector<Type> &left_types, uint64_t left_weight,
        std::vector<size_t> &interim_neighbors, std::vector<size_t> &type_counts, std::vector<Type> &type_values,
        std::vector<size_t> &destination_offsets, uint64_t right_weight, std::span<NodeId> out_neighbors,
        std::span<Type> out_edge_types, NodeId default_node_id, Type default_edge_type,
        boost::random::uniform_real_distribution<double> &toss, snark::Xoroshiro128PlusGenerator &gen) const;

    // Node features
    std::shared_ptr<BaseStorage<uint8_t>> m_node_features;
    std::vector<uint64_t> m_node_index;
    std::vector<uint64_t> m_node_feature_index;

    // Edge features
    std::shared_ptr<BaseStorage<uint8_t>> m_edge_features;
    std::vector<uint64_t> m_edge_feature_index;
    std::vector<uint64_t> m_edge_feature_offset;

    // Neighbor/edge indices
    std::vector<Type> m_edge_types;
    std::vector<uint64_t> m_edge_type_offset;
    std::vector<NodeId> m_edge_destination;
    std::vector<float> m_edge_weights;

    std::vector<uint64_t> m_neighbors_index;

    std::vector<Type> m_node_types;
    Metadata m_metadata;
    PartitionStorageType m_storage_type;
};

} // namespace snark
#endif
