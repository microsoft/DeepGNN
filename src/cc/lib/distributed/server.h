// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#ifndef SNARK_SERVER_H
#define SNARK_SERVER_H

#include <latch>
#include <memory>
#include <thread>

#include <grpc/grpc.h>
#include <grpcpp/create_channel.h>
#include <grpcpp/grpcpp.h>

#include "src/cc/lib/distributed/graph_engine.h"
#include "src/cc/lib/distributed/graph_sampler.h"
#include "src/cc/lib/graph/graph.h"

namespace snark
{
class GRPCServer final
{
  public:
    GRPCServer(std::shared_ptr<snark::GraphEngineServiceImpl> engine_service_impl,
               std::shared_ptr<snark::GraphSamplerServiceImpl> sampler_service_impl, std::string host_name,
               std::string ssl_key, std::string ssl_cert, std::string ssl_root);

    ~GRPCServer();

    std::shared_ptr<grpc::Channel> InProcessChannel();

    void HandleRpcs(size_t index);

  private:
    std::vector<std::unique_ptr<grpc::ServerCompletionQueue>> m_cqs;

    // Sampler/Engine split helps us to manage runtime:
    // * Resource heavy components can be deployed to separate machine types.
    // * Adding new server side samplers doesn't require to restart a service
    //   and interrupt existing clients, new clients can connect to old and new
    //   endpoints.
    snark::GraphEngine::AsyncService m_engine_service;
    std::shared_ptr<snark::GraphEngine::Service> m_engine_service_impl;
    snark::GraphSampler::AsyncService m_sampler_service;
    std::shared_ptr<snark::GraphSampler::Service> m_sampler_service_impl;
    std::unique_ptr<grpc::Server> m_server;
    std::vector<std::thread> m_runner_threads;
    std::atomic<bool> m_shutdown;
    std::latch m_latch;
};
} // namespace snark
#endif // SNARK_SERVER_H
