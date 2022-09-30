// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#ifndef SNARK_SERVICE_H
#define SNARK_SERVICE_H

#include "absl/container/flat_hash_map.h"
#include <grpc/grpc.h>
#include <grpcpp/channel.h>
#include <memory>

#include "src/cc/lib/distributed/service.grpc.pb.h"
#include "src/cc/lib/graph/graph.h"
#include "src/cc/lib/utils/threadpool.h"

namespace snark
{

class GraphEngineServiceImpl final : public snark::GraphEngine::Service
{
  public:
    GraphEngineServiceImpl(std::string path, std::vector<uint32_t> partitions, PartitionStorageType storage_type,
                           std::string config_path, bool enable_threadpool = false);
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

    // Split a bunch of items into several groups and for each group we use a
    // dedicated thread to process them.
    // Args:
    //  size: how many items will be split.
    //  preCallback: callback function to report how many tasks will be put into thread pool.
    //  callback: callback function for each individual task.
    //    index: index of the task.
    //    start_offset: start offset of the full node list.
    //    end_offset: end offset of the full node list
    void RunParallel(
        const std::size_t &size, std::function<void(const std::size_t &count)> preCallback,
        std::function<void(const std::size_t &index, const std::size_t &start_offset, const std::size_t &end_offset)>
            callback) const;

    std::vector<Partition> m_partitions;
    absl::flat_hash_map<NodeId, uint64_t> m_node_map;
    std::vector<uint32_t> m_partitions_indices;
    std::vector<uint64_t> m_internal_indices;
    std::vector<uint32_t> m_counts;
    Metadata m_metadata;
    std::shared_ptr<ThreadPool> m_threadPool;
};

} // namespace snark
#endif // SNARK_SERVICE_H
