// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#ifndef SNARK_SERVICE_H
#define SNARK_SERVICE_H

#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include <grpc/grpc.h>
#include <grpcpp/channel.h>
#include <memory>
#include <thread>

#include "boost/asio.hpp"
#include "src/cc/lib/distributed/service.grpc.pb.h"
#include "src/cc/lib/graph/graph.h"

namespace snark
{

class GraphEngineServiceImpl final : public snark::GraphEngine::Service
{
  public:
    GraphEngineServiceImpl(snark::Metadata metadata, std::vector<std::string> paths, std::vector<uint32_t> partitions,
                           PartitionStorageType storage_type);
    grpc::Status GetNodeTypes(::grpc::ServerContext *context, const snark::NodeTypesRequest *request,
                              snark::NodeTypesReply *response) override;

    grpc::Status GetNodeFeatures(::grpc::ServerContext *context, const snark::NodeFeaturesRequest *request,
                                 snark::NodeFeaturesReply *response) override;
    grpc::Status GetEdgeFeatures(::grpc::ServerContext *context, const snark::EdgeFeaturesRequest *request,
                                 snark::EdgeFeaturesReply *response) override;
    grpc::Status GetNodeSparseFeatures(::grpc::ServerContext *context, const snark::NodeSparseFeaturesRequest *request,
                                       snark::SparseFeaturesReply *response) override;
    grpc::Status GetEdgeSparseFeatures(::grpc::ServerContext *context, const snark::EdgeSparseFeaturesRequest *request,
                                       snark::SparseFeaturesReply *response) override;
    grpc::Status GetNodeStringFeatures(::grpc::ServerContext *context, const snark::NodeSparseFeaturesRequest *request,
                                       snark::StringFeaturesReply *response) override;
    grpc::Status GetEdgeStringFeatures(::grpc::ServerContext *context, const snark::EdgeSparseFeaturesRequest *request,
                                       snark::StringFeaturesReply *response) override;
    grpc::Status GetNeighborCounts(::grpc::ServerContext *context, const snark::GetNeighborsRequest *request,
                                   snark::GetNeighborCountsReply *response) override;
    grpc::Status GetNeighbors(::grpc::ServerContext *context, const snark::GetNeighborsRequest *request,
                              snark::GetNeighborsReply *response) override;
    grpc::Status WeightedSampleNeighbors(::grpc::ServerContext *context,
                                         const snark::WeightedSampleNeighborsRequest *request,
                                         snark::WeightedSampleNeighborsReply *response) override;
    grpc::Status UniformSampleNeighbors(::grpc::ServerContext *context,
                                        const snark::UniformSampleNeighborsRequest *request,
                                        snark::UniformSampleNeighborsReply *response) override;
    grpc::Status GetMetadata(::grpc::ServerContext *context, const snark::EmptyMessage *request,
                             snark::MetadataReply *response) override;

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
#endif // SNARK_SERVICE_H
