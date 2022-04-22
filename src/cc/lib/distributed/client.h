// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#ifndef SNARK_CLIENT_H
#define SNARK_CLIENT_H

#include <atomic>
#include <filesystem>
#include <functional>
#include <mutex>
#include <span>
#include <thread>

#include <grpc/grpc.h>
#include <grpcpp/channel.h>
#include <grpcpp/completion_queue.h>

#include "src/cc/lib/distributed/service.grpc.pb.h"
#include "src/cc/lib/graph/graph.h"

namespace snark
{

class GRPCClient final
{
  public:
    GRPCClient(std::vector<std::shared_ptr<grpc::Channel>> channels, uint32_t num_threads, uint32_t num_threads_per_cq);
    void GetNodeType(std::span<const NodeId> node_ids, std::span<Type> output, Type default_type);
    void GetNodeFeature(std::span<const NodeId> node_ids, std::span<FeatureMeta> features, std::span<uint8_t> output);

    void GetEdgeFeature(std::span<const NodeId> edge_src_ids, std::span<const NodeId> edge_dst_ids,
                        std::span<const Type> edge_types, std::span<FeatureMeta> features, std::span<uint8_t> output);

    void GetNodeSparseFeature(std::span<const NodeId> node_ids, std::span<const FeatureId> features,
                              std::span<int64_t> out_dimensions, std::vector<std::vector<int64_t>> &out_indices,
                              std::vector<std::vector<uint8_t>> &out_values);

    void GetEdgeSparseFeature(std::span<const NodeId> edge_src_ids, std::span<const NodeId> edge_dst_ids,
                              std::span<const Type> edge_types, std::span<const FeatureId> features,
                              std::span<int64_t> out_dimensions, std::vector<std::vector<int64_t>> &out_indices,
                              std::vector<std::vector<uint8_t>> &out_values);

    void FullNeighbor(std::span<const NodeId> node_ids, std::span<const Type> edge_types,
                      std::vector<NodeId> &output_nodes, std::vector<Type> &output_types,
                      std::vector<float> &output_weights, std::span<uint64_t> output_neighbor_counts);

    void WeightedSampleNeighbor(int64_t seed, std::span<const NodeId> node_ids, std::span<const Type> edge_types,
                                size_t count, std::span<NodeId> output_nodes, std::span<Type> output_types,
                                std::span<float> output_weights, NodeId default_node_id, float default_weight,
                                Type default_edge_type);

    void UniformSampleNeighbor(bool without_replacement, int64_t seed, std::span<const NodeId> node_ids,
                               std::span<const Type> edge_types, size_t count, std::span<NodeId> output_nodes,
                               std::span<Type> output_types, NodeId default_node_id, Type default_type);

    uint64_t CreateSampler(bool is_edge, CreateSamplerRequest_Category category, std::span<Type> types);

    void SampleNodes(int64_t seed, uint64_t sampler_id, std::span<NodeId> out_node_ids, std::span<Type> output_types);

    void SampleEdges(int64_t seed, uint64_t sampler_id, std::span<NodeId> out_src_node_ids,
                     std::span<Type> output_types, std::span<NodeId> out_dst_node_ids);
    void WriteMetadata(std::filesystem::path path);

    ~GRPCClient();

  private:
    std::mutex m_sampler_mutex;
    std::vector<std::vector<uint64_t>> m_sampler_ids;
    std::vector<std::vector<float>> m_sampler_weights;

    std::function<void()> AsyncCompleteRpc(size_t i);
    grpc::CompletionQueue *NextCompletionQueue();

    std::vector<std::unique_ptr<GraphEngine::Stub>> m_engine_stubs;
    std::vector<std::unique_ptr<GraphSampler::Stub>> m_sampler_stubs;
    std::vector<grpc::CompletionQueue> m_completion_queue;
    std::vector<std::thread> m_reply_threads;
    std::atomic<size_t> m_counter;
};

} // namespace snark
#endif // SNARK_CLIENT_H
