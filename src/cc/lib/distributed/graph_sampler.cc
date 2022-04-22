// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "src/cc/lib/distributed/graph_sampler.h"
#include "src/cc/lib/graph/sampler.h"
#include <cstdio>
#include <thread>

#include <glog/logging.h>
#include <glog/raw_logging.h>

namespace snark
{

GraphSamplerServiceImpl::GraphSamplerServiceImpl(std::string path, std::set<size_t> partitions)
    : m_partitions(std::move(partitions))
{
    m_node_sampler_factory[snark::CreateSamplerRequest_Category_WEIGHTED] =
        std::make_shared<WeightedNodeSamplerFactory>(path);
    m_node_sampler_factory[snark::CreateSamplerRequest_Category_UNIFORM_WITH_REPLACEMENT] =
        std::make_shared<UniformNodeSamplerFactory>(path);
    m_node_sampler_factory[snark::CreateSamplerRequest_Category_UNIFORM_WITHOUT_REPLACEMENT] =
        std::make_shared<UniformNodeSamplerFactoryWithoutReplacement>(path);
    m_edge_sampler_factory[snark::CreateSamplerRequest_Category_WEIGHTED] =
        std::make_shared<WeightedEdgeSamplerFactory>(path);
    m_edge_sampler_factory[snark::CreateSamplerRequest_Category_UNIFORM_WITH_REPLACEMENT] =
        std::make_shared<UniformEdgeSamplerFactory>(path);
    m_edge_sampler_factory[snark::CreateSamplerRequest_Category_UNIFORM_WITHOUT_REPLACEMENT] =
        std::make_shared<UniformEdgeSamplerFactoryWithoutReplacement>(path);
}

grpc::Status GraphSamplerServiceImpl::Create(::grpc::ServerContext *context, const snark::CreateSamplerRequest *request,
                                             snark::CreateSamplerReply *response)
{
    auto &factory = request->is_edge() ? m_edge_sampler_factory : m_node_sampler_factory;
    auto it = factory.find(request->category());
    if (it == std::end(factory))
    {
        RAW_LOG_ERROR("Failed to find sampler in path");
        return grpc::Status(grpc::StatusCode::FAILED_PRECONDITION, "Failed to find sampler in path");
    }

    auto sampler = it->second->Create(
        std::set<Type>(std::begin(request->enitity_types()), std::end(request->enitity_types())), m_partitions);
    if (!sampler)
    {
        return grpc::Status(grpc::StatusCode::FAILED_PRECONDITION, "Failed to create sampler");
    }

    response->set_weight(sampler->Weight());
    {
        std::lock_guard l(m_mutex);
        response->set_sampler_id(m_samplers.size());
        m_samplers.emplace_back(std::move(sampler));
    }

    return {};
}

grpc::Status GraphSamplerServiceImpl::Sample(::grpc::ServerContext *context, const snark::SampleRequest *request,
                                             snark::SampleReply *response)
{
    if (request->count() == 0)
    {
        return {};
    }

    snark::Sampler *sampler;
    {
        std::lock_guard l(m_mutex);
        sampler = m_samplers[request->sampler_id()].get();
    }

    const auto count = request->count();
    response->mutable_types()->Resize(count, {});
    if (request->is_edge())
    {
        response->mutable_node_ids()->Resize(2 * count, {});
        auto src = response->mutable_node_ids()->mutable_data();
        auto tp = response->mutable_types()->mutable_data();
        auto dst = src + count;
        sampler->Sample(request->seed(), std::span(tp, count), std::span(src, count), std::span(dst, count));
    }
    else
    {
        response->mutable_node_ids()->Resize(count, {});
        auto out = response->mutable_node_ids()->mutable_data();
        auto tp = response->mutable_types()->mutable_data();
        sampler->Sample(request->seed(), std::span(tp, count), std::span(out, count));
    }
    return {};
}
} // namespace snark
