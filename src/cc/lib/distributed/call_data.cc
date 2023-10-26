// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "src/cc/lib/distributed/call_data.h"
#include <atomic>

#include <memory>
#include <thread>

#include <grpc/grpc.h>
#include <grpcpp/grpcpp.h>

namespace snark
{
CallData::CallData(grpc::ServerCompletionQueue &cq) : m_cq(cq), m_status(CREATE)
{
}

NodeFeaturesCallData::NodeFeaturesCallData(GraphEngine::AsyncService &service, grpc::ServerCompletionQueue &cq,
                                           snark::GraphEngine::Service &service_impl)
    : CallData(cq), m_responder(&m_ctx), m_service_impl(service_impl), m_service(service)
{
    Proceed();
}

void NodeFeaturesCallData::Proceed()
{
    if (m_status == CREATE)
    {
        m_status = PROCESS;
        m_service.RequestGetNodeFeatures(&m_ctx, &m_request, &m_responder, &m_cq, &m_cq, this);
    }
    else if (m_status == PROCESS)
    {
        // All new objects will be deleted when we drain the request queue.
        new NodeFeaturesCallData(m_service, m_cq, m_service_impl);
        const auto status = m_service_impl.GetNodeFeatures(&m_ctx, &m_request, &m_reply);
        m_status = FINISH;
        m_responder.Finish(m_reply, status, this);
    }
    else
    {
        GPR_ASSERT(m_status == FINISH);
        delete this;
    }
}

UpdateNodeFeaturesCallData::UpdateNodeFeaturesCallData(GraphEngine::AsyncService &service,
                                                       grpc::ServerCompletionQueue &cq,
                                                       snark::GraphEngine::Service &service_impl)
    : CallData(cq), m_responder(&m_ctx), m_service_impl(service_impl), m_service(service)
{
    Proceed();
}

void UpdateNodeFeaturesCallData::Proceed()
{
    if (m_status == CREATE)
    {
        m_status = PROCESS;
        m_service.RequestUpdateNodeFeatures(&m_ctx, &m_request, &m_responder, &m_cq, &m_cq, this);
    }
    else if (m_status == PROCESS)
    {
        // All new objects will be deleted when we drain the request queue.
        new UpdateNodeFeaturesCallData(m_service, m_cq, m_service_impl);
        const auto status = m_service_impl.UpdateNodeFeatures(&m_ctx, &m_request, &m_reply);
        m_status = FINISH;
        m_responder.Finish(m_reply, status, this);
    }
    else
    {
        GPR_ASSERT(m_status == FINISH);
        delete this;
    }
}

EdgeFeaturesCallData::EdgeFeaturesCallData(GraphEngine::AsyncService &service, grpc::ServerCompletionQueue &cq,
                                           snark::GraphEngine::Service &service_impl)
    : CallData(cq), m_responder(&m_ctx), m_service_impl(service_impl), m_service(service)
{
    Proceed();
}

void EdgeFeaturesCallData::Proceed()
{
    if (m_status == CREATE)
    {
        m_status = PROCESS;
        m_service.RequestGetEdgeFeatures(&m_ctx, &m_request, &m_responder, &m_cq, &m_cq, this);
    }
    else if (m_status == PROCESS)
    {
        new EdgeFeaturesCallData(m_service, m_cq, m_service_impl);
        m_service_impl.GetEdgeFeatures(&m_ctx, &m_request, &m_reply);
        m_status = FINISH;
        m_responder.Finish(m_reply, grpc::Status::OK, this);
    }
    else
    {
        GPR_ASSERT(m_status == FINISH);
        delete this;
    }
}

GetNeighborCountCallData::GetNeighborCountCallData(GraphEngine::AsyncService &service, grpc::ServerCompletionQueue &cq,
                                                   snark::GraphEngine::Service &service_impl)
    : CallData(cq), m_responder(&m_ctx), m_service_impl(service_impl), m_service(service)

{
    Proceed();
}

void GetNeighborCountCallData::Proceed()
{
    if (m_status == CREATE)
    {
        m_status = PROCESS;
        m_service.RequestGetNeighborCounts(&m_ctx, &m_request, &m_responder, &m_cq, &m_cq, this);
    }
    else if (m_status == PROCESS)
    {
        new GetNeighborCountCallData(m_service, m_cq, m_service_impl);
        const auto status = m_service_impl.GetNeighborCounts(&m_ctx, &m_request, &m_reply);
        m_status = FINISH;
        m_responder.Finish(m_reply, status, this);
    }
    else
    {
        GPR_ASSERT(m_status == FINISH);
        delete this;
    }
}

GetNeighborsCallData::GetNeighborsCallData(GraphEngine::AsyncService &service, grpc::ServerCompletionQueue &cq,
                                           snark::GraphEngine::Service &service_impl)
    : CallData(cq), m_responder(&m_ctx), m_service_impl(service_impl), m_service(service)
{
    Proceed();
}

void GetNeighborsCallData::Proceed()
{
    if (m_status == CREATE)
    {
        m_status = PROCESS;
        m_service.RequestGetNeighbors(&m_ctx, &m_request, &m_responder, &m_cq, &m_cq, this);
    }
    else if (m_status == PROCESS)
    {
        new GetNeighborsCallData(m_service, m_cq, m_service_impl);
        const auto status = m_service_impl.GetNeighbors(&m_ctx, &m_request, &m_reply);
        m_status = FINISH;
        m_responder.Finish(m_reply, status, this);
    }
    else
    {
        GPR_ASSERT(m_status == FINISH);
        delete this;
    }
}

GetLastNCreatedNeighborCallData::GetLastNCreatedNeighborCallData(GraphEngine::AsyncService &service,
                                                                 grpc::ServerCompletionQueue &cq,
                                                                 snark::GraphEngine::Service &service_impl)
    : CallData(cq), m_responder(&m_ctx), m_service_impl(service_impl), m_service(service)
{
    Proceed();
}

void GetLastNCreatedNeighborCallData::Proceed()
{
    if (m_status == CREATE)
    {
        m_status = PROCESS;
        m_service.RequestGetLastNCreatedNeighbors(&m_ctx, &m_request, &m_responder, &m_cq, &m_cq, this);
    }
    else if (m_status == PROCESS)
    {
        new GetLastNCreatedNeighborCallData(m_service, m_cq, m_service_impl);
        const auto status = m_service_impl.GetLastNCreatedNeighbors(&m_ctx, &m_request, &m_reply);
        m_status = FINISH;
        m_responder.Finish(m_reply, status, this);
    }
    else
    {
        GPR_ASSERT(m_status == FINISH);
        delete this;
    }
}

SampleNeighborsCallData::SampleNeighborsCallData(GraphEngine::AsyncService &service, grpc::ServerCompletionQueue &cq,
                                                 snark::GraphEngine::Service &service_impl)
    : CallData(cq), m_responder(&m_ctx), m_service_impl(service_impl), m_service(service)
{
    Proceed();
}

void SampleNeighborsCallData::Proceed()
{
    if (m_status == CREATE)
    {
        m_status = PROCESS;
        m_service.RequestWeightedSampleNeighbors(&m_ctx, &m_request, &m_responder, &m_cq, &m_cq, this);
    }
    else if (m_status == PROCESS)
    {
        new SampleNeighborsCallData(m_service, m_cq, m_service_impl);
        const auto status = m_service_impl.WeightedSampleNeighbors(&m_ctx, &m_request, &m_reply);
        m_status = FINISH;
        m_responder.Finish(m_reply, status, this);
    }
    else
    {
        GPR_ASSERT(m_status == FINISH);
        delete this;
    }
}

UniformSampleNeighborsCallData::UniformSampleNeighborsCallData(GraphEngine::AsyncService &service,
                                                               grpc::ServerCompletionQueue &cq,
                                                               snark::GraphEngine::Service &service_impl)
    : CallData(cq), m_responder(&m_ctx), m_service_impl(service_impl), m_service(service)
{
    Proceed();
}

void UniformSampleNeighborsCallData::Proceed()
{
    if (m_status == CREATE)
    {
        m_status = PROCESS;
        m_service.RequestUniformSampleNeighbors(&m_ctx, &m_request, &m_responder, &m_cq, &m_cq, this);
    }
    else if (m_status == PROCESS)
    {
        new UniformSampleNeighborsCallData(m_service, m_cq, m_service_impl);
        const auto status = m_service_impl.UniformSampleNeighbors(&m_ctx, &m_request, &m_reply);
        m_status = FINISH;
        m_responder.Finish(m_reply, status, this);
    }
    else
    {
        GPR_ASSERT(m_status == FINISH);
        delete this;
    }
}

CreateSamplerCallData::CreateSamplerCallData(GraphSampler::AsyncService &service, grpc::ServerCompletionQueue &cq,
                                             snark::GraphSampler::Service &service_impl)
    : CallData(cq), m_responder(&m_ctx), m_service_impl(service_impl), m_service(service)
{
    Proceed();
}

void CreateSamplerCallData::Proceed()
{
    if (m_status == CREATE)
    {
        m_status = PROCESS;
        m_service.RequestCreate(&m_ctx, &m_request, &m_responder, &m_cq, &m_cq, this);
    }
    else if (m_status == PROCESS)
    {
        new CreateSamplerCallData(m_service, m_cq, m_service_impl);
        m_service_impl.Create(&m_ctx, &m_request, &m_reply);
        m_status = FINISH;
        m_responder.Finish(m_reply, grpc::Status::OK, this);
    }
    else
    {
        GPR_ASSERT(m_status == FINISH);
        delete this;
    }
}

SampleElementsCallData::SampleElementsCallData(GraphSampler::AsyncService &service, grpc::ServerCompletionQueue &cq,
                                               snark::GraphSampler::Service &service_impl)
    : CallData(cq), m_responder(&m_ctx), m_service_impl(service_impl), m_service(service)
{
    Proceed();
}

void SampleElementsCallData::Proceed()
{
    if (m_status == CREATE)
    {
        m_status = PROCESS;
        m_service.RequestSample(&m_ctx, &m_request, &m_responder, &m_cq, &m_cq, this);
    }
    else if (m_status == PROCESS)
    {
        new SampleElementsCallData(m_service, m_cq, m_service_impl);
        m_service_impl.Sample(&m_ctx, &m_request, &m_reply);
        m_status = FINISH;
        m_responder.Finish(m_reply, grpc::Status::OK, this);
    }
    else
    {
        GPR_ASSERT(m_status == FINISH);
        delete this;
    }
}

GetMetadataCallData::GetMetadataCallData(GraphEngine::AsyncService &service, grpc::ServerCompletionQueue &cq,
                                         snark::GraphEngine::Service &service_impl)
    : CallData(cq), m_responder(&m_ctx), m_service_impl(service_impl), m_service(service)
{
    Proceed();
}

void GetMetadataCallData::Proceed()
{
    if (m_status == CREATE)
    {
        m_status = PROCESS;
        m_service.RequestGetMetadata(&m_ctx, &m_request, &m_responder, &m_cq, &m_cq, this);
    }
    else if (m_status == PROCESS)
    {
        new GetMetadataCallData(m_service, m_cq, m_service_impl);
        m_service_impl.GetMetadata(&m_ctx, &m_request, &m_reply);
        m_status = FINISH;
        m_responder.Finish(m_reply, grpc::Status::OK, this);
    }
    else
    {
        GPR_ASSERT(m_status == FINISH);
        delete this;
    }
}

NodeTypesCallData::NodeTypesCallData(GraphEngine::AsyncService &service, grpc::ServerCompletionQueue &cq,
                                     snark::GraphEngine::Service &service_impl)
    : CallData(cq), m_responder(&m_ctx), m_service_impl(service_impl), m_service(service)
{
    Proceed();
}

void NodeTypesCallData::Proceed()
{
    if (m_status == CREATE)
    {
        m_status = PROCESS;
        m_service.RequestGetNodeTypes(&m_ctx, &m_request, &m_responder, &m_cq, &m_cq, this);
    }
    else if (m_status == PROCESS)
    {
        new NodeTypesCallData(m_service, m_cq, m_service_impl);
        const auto status = m_service_impl.GetNodeTypes(&m_ctx, &m_request, &m_reply);
        m_status = FINISH;
        m_responder.Finish(m_reply, status, this);
    }
    else
    {
        GPR_ASSERT(m_status == FINISH);
        delete this;
    }
}

NodeSparseFeaturesCallData::NodeSparseFeaturesCallData(GraphEngine::AsyncService &service,
                                                       grpc::ServerCompletionQueue &cq,
                                                       snark::GraphEngine::Service &service_impl)
    : CallData(cq), m_responder(&m_ctx), m_service_impl(service_impl), m_service(service)
{
    Proceed();
}

void NodeSparseFeaturesCallData::Proceed()
{
    if (m_status == CREATE)
    {
        m_status = PROCESS;
        m_service.RequestGetNodeSparseFeatures(&m_ctx, &m_request, &m_responder, &m_cq, &m_cq, this);
    }
    else if (m_status == PROCESS)
    {
        new NodeSparseFeaturesCallData(m_service, m_cq, m_service_impl);
        const auto status = m_service_impl.GetNodeSparseFeatures(&m_ctx, &m_request, &m_reply);
        m_status = FINISH;
        m_responder.Finish(m_reply, status, this);
    }
    else
    {
        GPR_ASSERT(m_status == FINISH);
        delete this;
    }
}

EdgeSparseFeaturesCallData::EdgeSparseFeaturesCallData(GraphEngine::AsyncService &service,
                                                       grpc::ServerCompletionQueue &cq,
                                                       snark::GraphEngine::Service &service_impl)
    : CallData(cq), m_responder(&m_ctx), m_service_impl(service_impl), m_service(service)
{
    Proceed();
}

void EdgeSparseFeaturesCallData::Proceed()
{
    if (m_status == CREATE)
    {
        m_status = PROCESS;
        m_service.RequestGetEdgeSparseFeatures(&m_ctx, &m_request, &m_responder, &m_cq, &m_cq, this);
    }
    else if (m_status == PROCESS)
    {
        new EdgeSparseFeaturesCallData(m_service, m_cq, m_service_impl);
        m_service_impl.GetEdgeSparseFeatures(&m_ctx, &m_request, &m_reply);
        m_status = FINISH;
        m_responder.Finish(m_reply, grpc::Status::OK, this);
    }
    else
    {
        GPR_ASSERT(m_status == FINISH);
        delete this;
    }
}

NodeStringFeaturesCallData::NodeStringFeaturesCallData(GraphEngine::AsyncService &service,
                                                       grpc::ServerCompletionQueue &cq,
                                                       snark::GraphEngine::Service &service_impl)
    : CallData(cq), m_responder(&m_ctx), m_service_impl(service_impl), m_service(service)
{
    Proceed();
}

void NodeStringFeaturesCallData::Proceed()
{
    if (m_status == CREATE)
    {
        m_status = PROCESS;
        m_service.RequestGetNodeStringFeatures(&m_ctx, &m_request, &m_responder, &m_cq, &m_cq, this);
    }
    else if (m_status == PROCESS)
    {
        new NodeStringFeaturesCallData(m_service, m_cq, m_service_impl);
        const auto status = m_service_impl.GetNodeStringFeatures(&m_ctx, &m_request, &m_reply);
        m_status = FINISH;
        m_responder.Finish(m_reply, status, this);
    }
    else
    {
        GPR_ASSERT(m_status == FINISH);
        delete this;
    }
}

EdgeStringFeaturesCallData::EdgeStringFeaturesCallData(GraphEngine::AsyncService &service,
                                                       grpc::ServerCompletionQueue &cq,
                                                       snark::GraphEngine::Service &service_impl)
    : CallData(cq), m_responder(&m_ctx), m_service_impl(service_impl), m_service(service)
{
    Proceed();
}

void EdgeStringFeaturesCallData::Proceed()
{
    if (m_status == CREATE)
    {
        m_status = PROCESS;
        m_service.RequestGetEdgeStringFeatures(&m_ctx, &m_request, &m_responder, &m_cq, &m_cq, this);
    }
    else if (m_status == PROCESS)
    {
        new EdgeStringFeaturesCallData(m_service, m_cq, m_service_impl);
        m_service_impl.GetEdgeStringFeatures(&m_ctx, &m_request, &m_reply);
        m_status = FINISH;
        m_responder.Finish(m_reply, grpc::Status::OK, this);
    }
    else
    {
        GPR_ASSERT(m_status == FINISH);
        delete this;
    }
}
} // namespace snark
