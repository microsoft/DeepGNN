// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#ifndef SNARK_GRAPH_SAMPLER_SERVICE_H
#define SNARK_GRAPH_SAMPLER_SERVICE_H

#include <mutex>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include <grpc/grpc.h>
#include <grpcpp/channel.h>

#include "src/cc/lib/distributed/service.grpc.pb.h"
#include "src/cc/lib/graph/graph.h"
#include "src/cc/lib/graph/logger.h"

namespace snark
{

// Stub value to indicate there are no items for a sampler
// and client is safe to skip requests to such shards.
const uint64_t empty_sampler_id = std::numeric_limits<uint64_t>::max();

class GraphSamplerServiceImpl final : public snark::GraphSampler::Service
{
  public:
    GraphSamplerServiceImpl(snark::Metadata metadata, std::vector<std::string> partition_paths,
                            std::vector<size_t> partition_indices, std::shared_ptr<Logger> logger = nullptr);

    grpc::Status Create(::grpc::ServerContext *context, const snark::CreateSamplerRequest *request,
                        snark::CreateSamplerReply *response) override;

    grpc::Status Sample(::grpc::ServerContext *context, const snark::SampleRequest *request,
                        snark::SampleReply *response) override;

  private:
    absl::flat_hash_map<snark::CreateSamplerRequest_Category, std::shared_ptr<snark::SamplerFactory>>
        m_node_sampler_factory;
    absl::flat_hash_map<snark::CreateSamplerRequest_Category, std::shared_ptr<snark::SamplerFactory>>
        m_edge_sampler_factory;
    std::vector<std::unique_ptr<snark::Sampler>> m_samplers;
    snark::Metadata m_metadata;
    std::vector<size_t> m_partition_indices;
    std::vector<std::string> m_patrition_paths;
    std::mutex m_mutex;
    std::shared_ptr<Logger> m_logger;
};

} // namespace snark
#endif // SNARK_GRAPH_SAMPLER_SERVICE_H
