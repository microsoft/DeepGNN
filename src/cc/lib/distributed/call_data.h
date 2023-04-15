// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#ifndef SNARK_CALL_DATA_H
#define SNARK_CALL_DATA_H

#include <memory>
#include <thread>

#include "src/cc/lib/distributed/graph_engine.h"
#include "src/cc/lib/distributed/graph_sampler.h"
#include "src/cc/lib/distributed/service.grpc.pb.h"

#include <grpc/grpc.h>
#include <grpcpp/grpcpp.h>

namespace snark
{

// Base class to handle requests.
class CallData
{
  public:
    CallData(grpc::ServerCompletionQueue &cq);

    virtual void Proceed() = 0;
    virtual ~CallData() = default;

  protected:
    grpc::ServerCompletionQueue &m_cq;
    grpc::ServerContext m_ctx;

    enum CallStatus
    {
        CREATE,
        PROCESS,
        FINISH
    };
    CallStatus m_status;
};

class NodeFeaturesCallData final : public CallData
{
  public:
    NodeFeaturesCallData(GraphEngine::AsyncService &service, grpc::ServerCompletionQueue &cq,
                         snark::GraphEngine::Service &service_impl);

    void Proceed() override;

  private:
    NodeFeaturesRequest m_request;
    NodeFeaturesReply m_reply;
    grpc::ServerAsyncResponseWriter<NodeFeaturesReply> m_responder;
    snark::GraphEngine::Service &m_service_impl;
    GraphEngine::AsyncService &m_service;
};

class EdgeFeaturesCallData final : public CallData
{
  public:
    EdgeFeaturesCallData(GraphEngine::AsyncService &service, grpc::ServerCompletionQueue &cq,
                         snark::GraphEngine::Service &service_impl);

    void Proceed() override;

  private:
    EdgeFeaturesRequest m_request;
    EdgeFeaturesReply m_reply;
    grpc::ServerAsyncResponseWriter<EdgeFeaturesReply> m_responder;
    snark::GraphEngine::Service &m_service_impl;
    GraphEngine::AsyncService &m_service;
};

class GetNeighborCountCallData final : public CallData
{
  public:
    GetNeighborCountCallData(GraphEngine::AsyncService &service, grpc::ServerCompletionQueue &cq,
                             snark::GraphEngine::Service &service_impl);

    void Proceed() override;

  private:
    GetNeighborsRequest m_request;
    GetNeighborCountsReply m_reply;
    grpc::ServerAsyncResponseWriter<GetNeighborCountsReply> m_responder;
    snark::GraphEngine::Service &m_service_impl;
    GraphEngine::AsyncService &m_service;
};

class GetNeighborsCallData final : public CallData
{
  public:
    GetNeighborsCallData(GraphEngine::AsyncService &service, grpc::ServerCompletionQueue &cq,
                         snark::GraphEngine::Service &service_impl);

    void Proceed() override;

  private:
    GetNeighborsRequest m_request;
    GetNeighborsReply m_reply;
    grpc::ServerAsyncResponseWriter<GetNeighborsReply> m_responder;
    snark::GraphEngine::Service &m_service_impl;
    GraphEngine::AsyncService &m_service;
};

class GetLastNCreatedNeighborCallData final : public CallData
{
  public:
    GetLastNCreatedNeighborCallData(GraphEngine::AsyncService &service, grpc::ServerCompletionQueue &cq,
                                    snark::GraphEngine::Service &service_impl);

    void Proceed() override;

  private:
    GetLastNCreatedNeighborsRequest m_request;
    GetNeighborsReply m_reply;
    grpc::ServerAsyncResponseWriter<GetNeighborsReply> m_responder;
    snark::GraphEngine::Service &m_service_impl;
    GraphEngine::AsyncService &m_service;
};

class SampleNeighborsCallData final : public CallData
{
  public:
    SampleNeighborsCallData(GraphEngine::AsyncService &service, grpc::ServerCompletionQueue &cq,
                            snark::GraphEngine::Service &service_impl);

    void Proceed() override;

  private:
    WeightedSampleNeighborsRequest m_request;
    WeightedSampleNeighborsReply m_reply;
    grpc::ServerAsyncResponseWriter<WeightedSampleNeighborsReply> m_responder;
    snark::GraphEngine::Service &m_service_impl;
    GraphEngine::AsyncService &m_service;
};

class UniformSampleNeighborsCallData final : public CallData
{
  public:
    UniformSampleNeighborsCallData(GraphEngine::AsyncService &service, grpc::ServerCompletionQueue &cq,
                                   snark::GraphEngine::Service &service_impl);

    void Proceed() override;

  private:
    UniformSampleNeighborsRequest m_request;
    UniformSampleNeighborsReply m_reply;
    grpc::ServerAsyncResponseWriter<UniformSampleNeighborsReply> m_responder;
    snark::GraphEngine::Service &m_service_impl;
    GraphEngine::AsyncService &m_service;
};

class CreateSamplerCallData final : public CallData
{
  public:
    CreateSamplerCallData(GraphSampler::AsyncService &service, grpc::ServerCompletionQueue &cq,
                          snark::GraphSampler::Service &service_impl);

    void Proceed() override;

  private:
    CreateSamplerRequest m_request;
    CreateSamplerReply m_reply;
    grpc::ServerAsyncResponseWriter<CreateSamplerReply> m_responder;
    snark::GraphSampler::Service &m_service_impl;
    GraphSampler::AsyncService &m_service;
};

class SampleElementsCallData final : public CallData
{
  public:
    SampleElementsCallData(GraphSampler::AsyncService &service, grpc::ServerCompletionQueue &cq,
                           snark::GraphSampler::Service &service_impl);

    void Proceed() override;

  private:
    SampleRequest m_request;
    SampleReply m_reply;
    grpc::ServerAsyncResponseWriter<SampleReply> m_responder;
    snark::GraphSampler::Service &m_service_impl;
    GraphSampler::AsyncService &m_service;
};

class GetMetadataCallData final : public CallData
{
  public:
    GetMetadataCallData(GraphEngine::AsyncService &service, grpc::ServerCompletionQueue &cq,
                        snark::GraphEngine::Service &service_impl);

    void Proceed() override;

  private:
    EmptyMessage m_request;
    MetadataReply m_reply;
    grpc::ServerAsyncResponseWriter<MetadataReply> m_responder;
    snark::GraphEngine::Service &m_service_impl;
    GraphEngine::AsyncService &m_service;
};

class NodeTypesCallData final : public CallData
{
  public:
    NodeTypesCallData(GraphEngine::AsyncService &service, grpc::ServerCompletionQueue &cq,
                      snark::GraphEngine::Service &service_impl);

    void Proceed() override;

  private:
    NodeTypesRequest m_request;
    NodeTypesReply m_reply;
    grpc::ServerAsyncResponseWriter<NodeTypesReply> m_responder;
    snark::GraphEngine::Service &m_service_impl;
    GraphEngine::AsyncService &m_service;
};

class NodeSparseFeaturesCallData final : public CallData
{
  public:
    NodeSparseFeaturesCallData(GraphEngine::AsyncService &service, grpc::ServerCompletionQueue &cq,
                               snark::GraphEngine::Service &service_impl);

    void Proceed() override;

  private:
    NodeSparseFeaturesRequest m_request;
    SparseFeaturesReply m_reply;
    grpc::ServerAsyncResponseWriter<SparseFeaturesReply> m_responder;
    snark::GraphEngine::Service &m_service_impl;
    GraphEngine::AsyncService &m_service;
};

class EdgeSparseFeaturesCallData final : public CallData
{
  public:
    EdgeSparseFeaturesCallData(GraphEngine::AsyncService &service, grpc::ServerCompletionQueue &cq,
                               snark::GraphEngine::Service &service_impl);

    void Proceed() override;

  private:
    EdgeSparseFeaturesRequest m_request;
    SparseFeaturesReply m_reply;
    grpc::ServerAsyncResponseWriter<SparseFeaturesReply> m_responder;
    snark::GraphEngine::Service &m_service_impl;
    GraphEngine::AsyncService &m_service;
};

class NodeStringFeaturesCallData final : public CallData
{
  public:
    NodeStringFeaturesCallData(GraphEngine::AsyncService &service, grpc::ServerCompletionQueue &cq,
                               snark::GraphEngine::Service &service_impl);

    void Proceed() override;

  private:
    NodeSparseFeaturesRequest m_request;
    StringFeaturesReply m_reply;
    grpc::ServerAsyncResponseWriter<StringFeaturesReply> m_responder;
    snark::GraphEngine::Service &m_service_impl;
    GraphEngine::AsyncService &m_service;
};

class EdgeStringFeaturesCallData final : public CallData
{
  public:
    EdgeStringFeaturesCallData(GraphEngine::AsyncService &service, grpc::ServerCompletionQueue &cq,
                               snark::GraphEngine::Service &service_impl);

    void Proceed() override;

  private:
    EdgeSparseFeaturesRequest m_request;
    StringFeaturesReply m_reply;
    grpc::ServerAsyncResponseWriter<StringFeaturesReply> m_responder;
    snark::GraphEngine::Service &m_service_impl;
    GraphEngine::AsyncService &m_service;
};
} // namespace snark
#endif
