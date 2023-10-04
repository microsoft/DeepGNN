// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "src/cc/lib/distributed/graph_sampler.h"
#include "src/cc/lib/graph/sampler.h"
#include <cstdio>
#include <thread>

namespace snark
{

GraphSamplerServiceImpl::GraphSamplerServiceImpl(snark::Metadata metadata, std::vector<std::string> partition_paths,
                                                 std::vector<size_t> partition_indices, std::shared_ptr<Logger> logger)
    : m_metadata(std::move(metadata)), m_partition_indices(std::move(partition_indices)),
      m_patrition_paths(std::move(partition_paths))
{
    if (!logger)
    {
        logger = std::make_shared<GLogger>();
    }
    m_logger = logger;
    if (m_patrition_paths.size() != m_partition_indices.size())
    {
        m_logger->log_fatal("Not enough %ld paths provided. Expected %ld for each alias table.",
                            m_patrition_paths.size(), m_partition_indices.size());
    }
    m_node_sampler_factory[snark::CreateSamplerRequest_Category_WEIGHTED] =
        std::make_shared<WeightedNodeSamplerFactory>(m_metadata, m_patrition_paths, m_partition_indices);
    m_node_sampler_factory[snark::CreateSamplerRequest_Category_UNIFORM_WITH_REPLACEMENT] =
        std::make_shared<UniformNodeSamplerFactory>(m_metadata, m_patrition_paths, m_partition_indices);
    m_node_sampler_factory[snark::CreateSamplerRequest_Category_UNIFORM_WITHOUT_REPLACEMENT] =
        std::make_shared<UniformNodeSamplerFactoryWithoutReplacement>(m_metadata, m_patrition_paths,
                                                                      m_partition_indices);
    m_edge_sampler_factory[snark::CreateSamplerRequest_Category_WEIGHTED] =
        std::make_shared<WeightedEdgeSamplerFactory>(m_metadata, m_patrition_paths, m_partition_indices);
    m_edge_sampler_factory[snark::CreateSamplerRequest_Category_UNIFORM_WITH_REPLACEMENT] =
        std::make_shared<UniformEdgeSamplerFactory>(m_metadata, m_patrition_paths, m_partition_indices);
    m_edge_sampler_factory[snark::CreateSamplerRequest_Category_UNIFORM_WITHOUT_REPLACEMENT] =
        std::make_shared<UniformEdgeSamplerFactoryWithoutReplacement>(m_metadata, m_patrition_paths,
                                                                      m_partition_indices);
}

grpc::Status GraphSamplerServiceImpl::Create(::grpc::ServerContext *context, const snark::CreateSamplerRequest *request,
                                             snark::CreateSamplerReply *response)
{
    auto &factory = request->is_edge() ? m_edge_sampler_factory : m_node_sampler_factory;
    auto it = factory.find(request->category());
    if (it == std::end(factory))
    {
        m_logger->log_error("Failed to find sampler in path");
        return grpc::Status(grpc::StatusCode::FAILED_PRECONDITION, "Failed to find sampler in path");
    }

    auto sampler =
        it->second->Create(std::set<Type>(std::begin(request->enitity_types()), std::end(request->enitity_types())));
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
