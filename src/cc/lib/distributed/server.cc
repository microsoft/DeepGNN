// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "server.h"

#include <cstdio>
#include <limits>

#include <glog/logging.h>
#include <glog/raw_logging.h>
#include <grpcpp/health_check_service_interface.h>

#include "src/cc/lib/distributed/call_data.h"

namespace snark
{

// Stubs to produce default values for the client.
// It is easier to handle corner cases via service implementation
// rather than processing exceptions in the transport layer.
class EmptyGraphSampler final : public snark::GraphSampler::Service
{
  public:
    grpc::Status Create(::grpc::ServerContext *context, const snark::CreateSamplerRequest *request,
                        snark::CreateSamplerReply *response) override
    {
        response->set_weight(0.0f);
        response->set_sampler_id(empty_sampler_id);
        return grpc::Status::OK;
    }

    grpc::Status Sample(::grpc::ServerContext *context, const snark::SampleRequest *request,
                        snark::SampleReply *response) override
    {
        RAW_LOG_ERROR("Received request to an empty sampler");
        return grpc::Status(grpc::StatusCode::FAILED_PRECONDITION,
                            "Empty sampler should be filtered on the client side.");
    }
};

class EmptyGraphEngine final : public snark::GraphEngine::Service
{
  public:
    grpc::Status GetNodeTypes(::grpc::ServerContext *context, const snark::NodeTypesRequest *request,
                              snark::NodeTypesReply *response) override
    {
        return grpc::Status::OK;
    }

    grpc::Status GetNodeFeatures(::grpc::ServerContext *context, const snark::NodeFeaturesRequest *request,
                                 snark::NodeFeaturesReply *response) override
    {
        return grpc::Status::OK;
    }

    grpc::Status GetEdgeFeatures(::grpc::ServerContext *context, const snark::EdgeFeaturesRequest *request,
                                 snark::EdgeFeaturesReply *response) override
    {
        return grpc::Status::OK;
    }

    grpc::Status GetNodeSparseFeatures(::grpc::ServerContext *context, const snark::NodeSparseFeaturesRequest *request,
                                       snark::SparseFeaturesReply *response) override
    {
        return grpc::Status::OK;
    }

    grpc::Status GetEdgeSparseFeatures(::grpc::ServerContext *context, const snark::EdgeSparseFeaturesRequest *request,
                                       snark::SparseFeaturesReply *response) override
    {
        return grpc::Status::OK;
    }

    grpc::Status GetNodeStringFeatures(::grpc::ServerContext *context, const snark::NodeSparseFeaturesRequest *request,
                                       snark::StringFeaturesReply *response) override
    {
        return grpc::Status::OK;
    }

    grpc::Status GetEdgeStringFeatures(::grpc::ServerContext *context, const snark::EdgeSparseFeaturesRequest *request,
                                       snark::StringFeaturesReply *response) override
    {
        return grpc::Status::OK;
    }

    grpc::Status GetNeighborCount (::grpc::ServerContext *context, const snark::GetNeighborsRequest *request,
                                   snark::GetNeighborCountsReply *response) override
    {
        return grpc::Status::OK;
    }

    grpc::Status GetNeighbors(::grpc::ServerContext *context, const snark::GetNeighborsRequest *request,
                              snark::GetNeighborsReply *response) override
    {
        return grpc::Status::OK;
    }

    grpc::Status WeightedSampleNeighbors(::grpc::ServerContext *context,
                                         const snark::WeightedSampleNeighborsRequest *request,
                                         snark::WeightedSampleNeighborsReply *response) override
    {
        return grpc::Status::OK;
    }

    grpc::Status UniformSampleNeighbors(::grpc::ServerContext *context,
                                        const snark::UniformSampleNeighborsRequest *request,
                                        snark::UniformSampleNeighborsReply *response) override
    {
        return grpc::Status::OK;
    }
};

GRPCServer::GRPCServer(std::shared_ptr<snark::GraphEngineServiceImpl> engine_service_impl,
                       std::shared_ptr<snark::GraphSamplerServiceImpl> sampler_service_impl, std::string host_name,
                       std::string ssl_key, std::string ssl_cert, std::string ssl_root)
    : m_engine_service_impl(std::move(engine_service_impl)), m_sampler_service_impl(std::move(sampler_service_impl))
{
    if (!m_engine_service_impl && !m_sampler_service_impl)
    {
        RAW_LOG_FATAL("No services to start");
    }

    grpc::EnableDefaultHealthCheckService(true);
    grpc::ServerBuilder builder;
    auto creds = grpc::InsecureServerCredentials();
    if (!ssl_key.empty())
    {
        grpc::SslServerCredentialsOptions opts;
        opts.pem_root_certs = ssl_root;
        opts.pem_key_cert_pairs = {{ssl_key, ssl_cert}};
        creds = grpc::SslServerCredentials(opts);
    }

    builder.AddListeningPort(host_name, std::move(creds));
    if (!m_engine_service_impl)
    {
        m_engine_service_impl = std::make_shared<EmptyGraphEngine>();
    }
    builder.RegisterService(&m_engine_service);

    if (!m_sampler_service_impl)
    {
        m_sampler_service_impl = std::make_shared<EmptyGraphSampler>();
    }
    builder.RegisterService(&m_sampler_service);

    for (size_t thread_num = 0; thread_num < std::thread::hardware_concurrency(); ++thread_num)
    {
        m_cqs.emplace_back(builder.AddCompletionQueue());
    }

    m_server = builder.BuildAndStart();
    for (size_t thread_num = 0; thread_num < std::thread::hardware_concurrency(); ++thread_num)
    {
        m_runner_threads.emplace_back(&GRPCServer::HandleRpcs, this, thread_num);
    }
}

GRPCServer::~GRPCServer()
{
    m_server->Shutdown();
    for (auto &queue : m_cqs)
    {
        queue->Shutdown();
    }
    for (auto &thread : m_runner_threads)
    {
        thread.join();
    }
}

std::shared_ptr<grpc::Channel> GRPCServer::InProcessChannel()
{
    return m_server->InProcessChannel(grpc::ChannelArguments());
}

void GRPCServer::HandleRpcs(size_t index)
{
    auto &queue = *m_cqs[index];
    if (m_engine_service_impl)
    {
        new GetNeighborsCallData(m_engine_service, queue, *m_engine_service_impl);
        new SampleNeighborsCallData(m_engine_service, queue, *m_engine_service_impl);
        new UniformSampleNeighborsCallData(m_engine_service, queue, *m_engine_service_impl);
        new NodeFeaturesCallData(m_engine_service, queue, *m_engine_service_impl);
        new EdgeFeaturesCallData(m_engine_service, queue, *m_engine_service_impl);
        new NodeSparseFeaturesCallData(m_engine_service, queue, *m_engine_service_impl);
        new EdgeSparseFeaturesCallData(m_engine_service, queue, *m_engine_service_impl);
        new NodeStringFeaturesCallData(m_engine_service, queue, *m_engine_service_impl);
        new EdgeStringFeaturesCallData(m_engine_service, queue, *m_engine_service_impl);
        new GetMetadataCallData(m_engine_service, queue, *m_engine_service_impl);
        new NodeTypesCallData(m_engine_service, queue, *m_engine_service_impl);
    }
    if (m_sampler_service_impl)
    {
        new CreateSamplerCallData(m_sampler_service, queue, *m_sampler_service_impl);
        new SampleElementsCallData(m_sampler_service, queue, *m_sampler_service_impl);
    }

    void *tag;
    bool ok;
    while (queue.Next(&tag, &ok))
    {
        if (!ok)
        {
            // Keep draining queue for all call data types.
            delete static_cast<CallData *>(tag);
            continue;
        }

        static_cast<CallData *>(tag)->Proceed();
    }
}

} // namespace snark
