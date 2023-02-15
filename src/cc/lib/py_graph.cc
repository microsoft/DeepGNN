// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "py_graph.h"

#include <algorithm>
#include <cstdint>
#include <exception>
#include <functional>
#include <future>
#include <memory>
#include <queue>
#include <random>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef SNARK_PLATFORM_LINUX
// We need to include this file exactly once to override all malloc references in other files.
#include <mimalloc-override.h>
#endif

#include "absl/container/flat_hash_map.h"
#include "boost/random/uniform_real_distribution.hpp"

#include <grpcpp/create_channel.h>
// Use raw log to avoid possible initialization conflicts with glog from other libraries.
#include <glog/logging.h>
#include <glog/raw_logging.h>

#include "distributed/client.h"
#include "distributed/graph_engine.h"
#include "distributed/graph_sampler.h"
#include "graph/graph.h"
#include "graph/xoroshiro.h"

namespace deep_graph
{
namespace python
{

enum SamplerType
{
    Weighted,
    Uniform,
    UniformWithoutReplacement,

    Last
};

namespace
{
absl::flat_hash_map<SamplerType, snark::CreateSamplerRequest_Category> localToRemoteSamplerType = {
    {Weighted, snark::CreateSamplerRequest_Category::CreateSamplerRequest_Category_WEIGHTED},
    {Uniform, snark::CreateSamplerRequest_Category::CreateSamplerRequest_Category_UNIFORM_WITH_REPLACEMENT},
    {UniformWithoutReplacement,
     snark::CreateSamplerRequest_Category::CreateSamplerRequest_Category_UNIFORM_WITHOUT_REPLACEMENT},
};
} // namespace

struct GraphInternal
{
    std::unique_ptr<snark::Graph> graph;
    absl::flat_hash_map<SamplerType, std::shared_ptr<snark::SamplerFactory>> node_sampler_factory;
    absl::flat_hash_map<SamplerType, std::shared_ptr<snark::SamplerFactory>> edge_sampler_factory;
    std::shared_ptr<snark::GRPCClient> client;
};

template <bool is_node> class RemoteSampler final : public snark::Sampler
{
  public:
    RemoteSampler(SamplerType samplerType, size_t count, int32_t *types, std::shared_ptr<snark::GRPCClient> client);
    void Sample(int64_t seed, std::span<snark::Type> out_types, std::span<snark::NodeId> out_nodes, ...) const override;
    float Weight() const override;

  private:
    std::shared_ptr<snark::GRPCClient> m_client;
    uint64_t m_sampler_id;
};

template <bool is_node>
RemoteSampler<is_node>::RemoteSampler(SamplerType samplerType, size_t count, int32_t *types,
                                      std::shared_ptr<snark::GRPCClient> client)
    : m_client(std::move(client))
{
    m_sampler_id = m_client->CreateSampler(!is_node, localToRemoteSamplerType[samplerType],
                                           std::span<snark::Type>(types, types + count));
}

template <>
void RemoteSampler<false>::Sample(int64_t seed, std::span<snark::Type> out_types, std::span<snark::NodeId> out_nodes,
                                  ...) const
{
    va_list args;
    va_start(args, out_nodes);
    auto out_dst = va_arg(args, std::span<snark::NodeId>);
    m_client->SampleEdges(seed, m_sampler_id, out_nodes, out_types, out_dst);
    va_end(args);
}

template <>
void RemoteSampler<true>::Sample(int64_t seed, std::span<snark::Type> out_types, std::span<snark::NodeId> out_nodes,
                                 ...) const
{
    m_client->SampleNodes(seed, m_sampler_id, out_nodes, out_types);
}

template <bool is_node> float RemoteSampler<is_node>::Weight() const
{
    // We are sampling from the whole graph, so it doesn't really matter what weight is it in the interface.
    // If we start to use multiple clients, then we'll need to calculate an aggregate weight of all nodes/edges in the
    // graph per client.
    return 1.0f;
}

template <SamplerType samplerType, bool is_node>
int create_sampler(PyGraph *py_graph, PySampler *py_sampler, size_t count, int32_t *types)
{
    // We don't need to check python pointers because client wrapper always
    // create new objects and never explicitly deletes them(which happens with GC)
    auto &internal_graph = py_graph->graph;
    if (internal_graph == nullptr)
    {
        RAW_LOG_ERROR("Python graph is not initialized");
        return 1;
    }

    if (py_graph->graph->client)
    {
        try
        {
            py_sampler->sampler =
                std::make_unique<RemoteSampler<is_node>>(samplerType, count, types, py_graph->graph->client);
            return 0;
        }
        catch (const std::exception &e)
        {
            RAW_LOG_ERROR("Exception while creating sampler: %s", e.what());
            return 1;
        }
    }

    auto &factory = is_node ? internal_graph->node_sampler_factory : internal_graph->edge_sampler_factory;
    py_sampler->sampler = factory[samplerType]->Create(std::set<snark::Type>(types, types + count));
    if (py_sampler->sampler == nullptr)
    {
        RAW_LOG_ERROR("Failed to create %s: sampler", (is_node ? "node" : "edge"));
        return 1;
    }

    return 0;
}

int32_t CreateWeightedNodeSampler(PyGraph *py_graph, PySampler *node_sampler, size_t count, int32_t *types)
{
    return create_sampler<SamplerType::Weighted, true>(py_graph, node_sampler, count, types);
}

int32_t CreateUniformNodeSampler(PyGraph *py_graph, PySampler *node_sampler, size_t count, int32_t *types)
{
    return create_sampler<SamplerType::Uniform, true>(py_graph, node_sampler, count, types);
}

int32_t CreateUniformNodeSamplerWithoutReplacement(PyGraph *py_graph, PySampler *node_sampler, size_t count,
                                                   int32_t *types)
{
    return create_sampler<SamplerType::UniformWithoutReplacement, true>(py_graph, node_sampler, count, types);
}

int32_t SampleNodes(PySampler *py_sampler, int64_t seed, size_t count, NodeID *out_nodes, Type *out_types)
{
    if (py_sampler->sampler == nullptr)
    {
        RAW_LOG_ERROR("Internal node sampler is not initialized");
        return 1;
    }

    try
    {
        py_sampler->sampler->Sample(seed, std::span(reinterpret_cast<snark::Type *>(out_types), count),
                                    std::span(reinterpret_cast<snark::NodeId *>(out_nodes), count));

        return 0;
    }
    catch (const std::exception &e)
    {
        RAW_LOG_ERROR("Exception while fetching features: %s", e.what());
        return 1;
    }
}

int32_t CreateWeightedEdgeSampler(PyGraph *py_graph, PySampler *edge_sampler, size_t count, int32_t *types)
{
    return create_sampler<SamplerType::Weighted, false>(py_graph, edge_sampler, count, types);
}

int32_t CreateUniformEdgeSampler(PyGraph *py_graph, PySampler *edge_sampler, size_t count, int32_t *types)
{
    return create_sampler<SamplerType::Uniform, false>(py_graph, edge_sampler, count, types);
}

int32_t CreateUniformEdgeSamplerWithoutReplacement(PyGraph *py_graph, PySampler *edge_sampler, size_t count,
                                                   int32_t *types)
{
    return create_sampler<SamplerType::UniformWithoutReplacement, false>(py_graph, edge_sampler, count, types);
}

int32_t SampleEdges(PySampler *py_sampler, int64_t seed, size_t count, NodeID *out_src_id, NodeID *out_dst_id,
                    Type *out_type)
{
    if (py_sampler->sampler == nullptr)
    {
        RAW_LOG_ERROR("Internal edge sampler is not initialized");
        return 1;
    }
    try
    {
        py_sampler->sampler->Sample(seed, std::span(reinterpret_cast<snark::Type *>(out_type), count),
                                    std::span(reinterpret_cast<snark::NodeId *>(out_src_id), count),
                                    std::span(reinterpret_cast<snark::NodeId *>(out_dst_id), count));
        return 0;
    }
    catch (const std::exception &e)
    {
        RAW_LOG_ERROR("Exception while sampling edges: %s", e.what());
        return 1;
    }
}

int32_t CreateLocalGraph(PyGraph *py_graph, const char *meta_location, size_t count, uint32_t *partitions,
                         const char **partition_locations, PyPartitionStorageType storage_type_,
                         const char *config_path)
{
    snark::PartitionStorageType storage_type = static_cast<snark::PartitionStorageType>(storage_type_);
    snark::Metadata metadata(meta_location, config_path);

    std::vector<std::string> partition_paths;
    partition_paths.reserve(count);
    for (size_t i = 0; i < count; ++i)
    {
        partition_paths.emplace_back(partition_locations[i]);
    }

    py_graph->graph = std::make_unique<GraphInternal>();
    std::vector<size_t> partition_indices(partitions, partitions + count);
    py_graph->graph->graph = std::make_unique<snark::Graph>(
        metadata, partition_paths, std::vector<uint32_t>(partitions, partitions + count), storage_type);
    py_graph->graph->node_sampler_factory[SamplerType::Weighted] =
        std::make_shared<snark::WeightedNodeSamplerFactory>(metadata, partition_paths, partition_indices);
    py_graph->graph->node_sampler_factory[SamplerType::Uniform] =
        std::make_shared<snark::UniformNodeSamplerFactory>(metadata, partition_paths, partition_indices);
    py_graph->graph->node_sampler_factory[SamplerType::UniformWithoutReplacement] =
        std::make_shared<snark::UniformNodeSamplerFactoryWithoutReplacement>(metadata, partition_paths,
                                                                             partition_indices);

    py_graph->graph->edge_sampler_factory[SamplerType::Weighted] =
        std::make_shared<snark::WeightedEdgeSamplerFactory>(metadata, partition_paths, partition_indices);
    py_graph->graph->edge_sampler_factory[SamplerType::Uniform] =
        std::make_shared<snark::UniformEdgeSamplerFactory>(metadata, partition_paths, partition_indices);
    py_graph->graph->edge_sampler_factory[SamplerType::UniformWithoutReplacement] =
        std::make_shared<snark::UniformEdgeSamplerFactoryWithoutReplacement>(metadata, std::move(partition_paths),
                                                                             std::move(partition_indices));
    if (py_graph->graph == nullptr)
    {
        RAW_LOG_ERROR("Internal graph wasn't initialized");
        return 1;
    }

    return 0;
}

int32_t CreateRemoteClient(PyGraph *py_graph, const char *output_folder, const char **connection,
                           size_t connection_count, const char *ssl_cert, size_t num_threads, size_t num_threads_per_cq,
                           size_t num_custom_args, const char **custom_args_keys, const char **custom_args_values)
{
    py_graph->graph = std::make_unique<GraphInternal>();
    std::vector<std::shared_ptr<grpc::Channel>> channels;
    auto creds = grpc::InsecureChannelCredentials();
    if (ssl_cert != nullptr && strlen(ssl_cert) > 0)
    {
        grpc::SslCredentialsOptions ssl_opts;
        ssl_opts.pem_root_certs = ssl_cert;
        creds = grpc::SslCredentials(ssl_opts);
    }
    grpc::ChannelArguments args;

    args.SetMaxReceiveMessageSize(-1);
    for (size_t custom_arg_index = 0; custom_arg_index < num_custom_args; ++custom_arg_index)
    {
        try
        {
            auto val = std::stoi(std::string(custom_args_values[custom_arg_index]));
            args.SetInt(custom_args_keys[custom_arg_index], val);
        }
        catch (std::invalid_argument const &ex)
        {
            args.SetString(custom_args_keys[custom_arg_index], custom_args_values[custom_arg_index]);
        }
    }

    for (size_t i = 0; i < connection_count; ++i)
    {
        channels.emplace_back(grpc::CreateCustomChannel(connection[i], creds, args));
    }

    py_graph->graph->client =
        std::make_unique<snark::GRPCClient>(std::move(channels), uint32_t(num_threads), uint32_t(num_threads_per_cq));
    py_graph->graph->client->WriteMetadata(output_folder);
    return 0;
}

std::vector<snark::FeatureMeta> ExtractFeatureInfo(Feature *features, size_t features_size)
{
    std::vector<snark::FeatureMeta> features_info;
    features_info.reserve(features_size);
    for (size_t index = 0; index < features_size; ++index)
    {
        snark::FeatureId id = *features;
        ++features;
        snark::FeatureSize dim = *features;
        ++features;
        features_info.emplace_back(id, dim);
    }

    return features_info;
}

int32_t GetNodeType(PyGraph *py_graph, NodeID *node_ids, size_t node_ids_size, Type *output, Type default_type)
{
    if (py_graph->graph == nullptr)
    {
        RAW_LOG_ERROR("Internal graph is not initialized");
        return 1;
    }
    if (py_graph->graph->graph)
    {
        py_graph->graph->graph->GetNodeType(std::span(reinterpret_cast<snark::NodeId *>(node_ids), node_ids_size),
                                            std::span(reinterpret_cast<snark::Type *>(output), node_ids_size),
                                            default_type);
        return 0;
    }

    try
    {
        py_graph->graph->client->GetNodeType(std::span(reinterpret_cast<snark::NodeId *>(node_ids), node_ids_size),
                                             std::span(reinterpret_cast<snark::Type *>(output), node_ids_size),
                                             default_type);
        return 0;
    }
    catch (const std::exception &e)
    {
        RAW_LOG_ERROR("Exception while fetching node features: %s", e.what());
        return 1;
    }
}

int32_t GetNodeFeature(PyGraph *py_graph, NodeID *node_ids, size_t node_ids_size, Feature *features,
                       size_t features_size, uint8_t *output, size_t output_size)
{
    if (py_graph->graph == nullptr)
    {
        RAW_LOG_ERROR("Internal graph is not initialized");
        return 1;
    }

    auto features_info = ExtractFeatureInfo(features, features_size);
    if (py_graph->graph->graph)
    {
        py_graph->graph->graph->GetNodeFeature(std::span(reinterpret_cast<snark::NodeId *>(node_ids), node_ids_size),
                                               std::span(features_info), std::span(output, output_size));
        return 0;
    }

    try
    {
        py_graph->graph->client->GetNodeFeature(std::span(reinterpret_cast<snark::NodeId *>(node_ids), node_ids_size),
                                                std::span(features_info),
                                                std::span(reinterpret_cast<uint8_t *>(output), output_size));
        return 0;
    }
    catch (const std::exception &e)
    {
        RAW_LOG_ERROR("Exception while fetching node features: %s", e.what());
        return 1;
    }
}

int32_t GetNodeSparseFeature(PyGraph *py_graph, NodeID *node_ids, size_t node_ids_size, Feature *features,
                             size_t features_size, GetSparseFeaturesCallback callback)
{
    if (py_graph->graph == nullptr)
    {
        RAW_LOG_ERROR("Internal graph is not initialized");
        return 1;
    }

    std::vector<std::vector<int64_t>> indices(features_size);
    std::vector<std::vector<uint8_t>> data(features_size);
    std::vector<int64_t> dimensions(features_size);

    if (py_graph->graph->graph)
    {
        py_graph->graph->graph->GetNodeSparseFeature(
            std::span(reinterpret_cast<snark::NodeId *>(node_ids), node_ids_size), std::span(features, features_size),
            std::span(dimensions), indices, data);
    }
    else
    {
        try
        {
            py_graph->graph->client->GetNodeSparseFeature(
                std::span(reinterpret_cast<snark::NodeId *>(node_ids), node_ids_size),
                std::span(features, features_size), std::span(dimensions), indices, data);
        }
        catch (const std::exception &e)
        {
            RAW_LOG_ERROR("Exception while fetching node features: %s", e.what());
            return 1;
        }
    }

    // Pointers for python to copy data from C++.
    std::vector<const int64_t *> indices_ptrs;
    std::vector<size_t> indices_sizes;
    std::vector<const uint8_t *> data_ptrs;
    std::vector<size_t> data_sizes;

    for (size_t i = 0; i < features_size; ++i)
    {
        indices_ptrs.emplace_back(indices[i].data());
        indices_sizes.emplace_back(indices[i].size());
        data_ptrs.emplace_back(data[i].data());
        data_sizes.emplace_back(data[i].size());
    }

    callback(indices_ptrs.data(), indices_sizes.data(), data_ptrs.data(), data_sizes.data(), dimensions.data());
    return 0;
}

int32_t GetNodeStringFeature(PyGraph *py_graph, NodeID *node_ids, size_t node_ids_size, Feature *features,
                             size_t features_size, int64_t *dimensions, GetStringFeaturesCallback callback)
{
    if (py_graph->graph == nullptr)
    {
        RAW_LOG_ERROR("Internal graph is not initialized");
        return 1;
    }

    std::vector<uint8_t> data;
    if (py_graph->graph->graph)
    {
        py_graph->graph->graph->GetNodeStringFeature(
            std::span(reinterpret_cast<snark::NodeId *>(node_ids), node_ids_size), std::span(features, features_size),
            std::span(dimensions, node_ids_size * features_size), data);
    }
    else
    {
        try
        {
            py_graph->graph->client->GetNodeStringFeature(
                std::span(reinterpret_cast<snark::NodeId *>(node_ids), node_ids_size),
                std::span(features, features_size), std::span(dimensions, node_ids_size * features_size), data);
        }
        catch (const std::exception &e)
        {
            RAW_LOG_ERROR("Exception while fetching node features: %s", e.what());
            return 1;
        }
    }

    callback(data.size(), data.data());
    return 0;
}

int32_t GetEdgeFeature(PyGraph *py_graph, NodeID *edge_src_ids, NodeID *edge_dst_ids, Type *edge_types,
                       size_t edges_size, Feature *features, size_t features_size, uint8_t *output, size_t output_size)
{
    if (py_graph->graph == nullptr)
    {
        RAW_LOG_ERROR("Internal graph is not initialized");
        return 1;
    }

    auto features_info = ExtractFeatureInfo(features, features_size);
    if (py_graph->graph->graph)
    {
        py_graph->graph->graph->GetEdgeFeature(std::span(reinterpret_cast<snark::NodeId *>(edge_src_ids), edges_size),
                                               std::span(reinterpret_cast<snark::NodeId *>(edge_dst_ids), edges_size),
                                               std::span(reinterpret_cast<snark::Type *>(edge_types), edges_size),
                                               std::span(features_info), std::span(output, output_size));
        return 0;
    }

    try
    {
        py_graph->graph->client->GetEdgeFeature(std::span(reinterpret_cast<snark::NodeId *>(edge_src_ids), edges_size),
                                                std::span(reinterpret_cast<snark::NodeId *>(edge_dst_ids), edges_size),
                                                std::span(reinterpret_cast<snark::Type *>(edge_types), edges_size),
                                                std::span(features_info), std::span(output, output_size));
        return 0;
    }
    catch (const std::exception &e)
    {
        RAW_LOG_ERROR("Exception while fetching edge features: %s", e.what());
        return 1;
    }
}

int32_t GetEdgeSparseFeature(PyGraph *py_graph, NodeID *edge_src_ids, NodeID *edge_dst_ids, Type *edge_types,
                             size_t edges_size, Feature *features, size_t features_size,
                             GetSparseFeaturesCallback callback)
{
    if (py_graph->graph == nullptr)
    {
        RAW_LOG_ERROR("Internal graph is not initialized");
        return 1;
    }

    std::vector<std::vector<int64_t>> indices(features_size);
    std::vector<std::vector<uint8_t>> data(features_size);
    std::vector<int64_t> dimensions(features_size);

    // Pointers for python to copy data from C++.
    std::vector<const int64_t *> indices_ptrs;
    std::vector<size_t> indices_sizes;
    std::vector<const uint8_t *> data_ptrs;
    std::vector<size_t> data_sizes;

    if (py_graph->graph->graph)
    {
        py_graph->graph->graph->GetEdgeSparseFeature(
            std::span(reinterpret_cast<snark::NodeId *>(edge_src_ids), edges_size),
            std::span(reinterpret_cast<snark::NodeId *>(edge_dst_ids), edges_size),
            std::span(reinterpret_cast<snark::Type *>(edge_types), edges_size), std::span(features, features_size),
            std::span(dimensions), indices, data);
    }

    else
    {
        try
        {
            py_graph->graph->client->GetEdgeSparseFeature(
                std::span(reinterpret_cast<snark::NodeId *>(edge_src_ids), edges_size),
                std::span(reinterpret_cast<snark::NodeId *>(edge_dst_ids), edges_size),
                std::span(reinterpret_cast<snark::Type *>(edge_types), edges_size), std::span(features, features_size),
                std::span(dimensions), indices, data);
        }
        catch (const std::exception &e)
        {
            RAW_LOG_ERROR("Exception while fetching node features: %s", e.what());
            return 1;
        }
    }

    for (size_t i = 0; i < features_size; ++i)
    {
        indices_ptrs.emplace_back(indices[i].data());
        indices_sizes.emplace_back(indices[i].size());
        data_ptrs.emplace_back(data[i].data());
        data_sizes.emplace_back(data[i].size());
    }

    callback(indices_ptrs.data(), indices_sizes.data(), data_ptrs.data(), data_sizes.data(), dimensions.data());
    return 0;
}

int32_t GetEdgeStringFeature(PyGraph *py_graph, NodeID *edge_src_ids, NodeID *edge_dst_ids, Type *edge_types,
                             size_t edge_size, Feature *features, size_t features_size, int64_t *dimensions,
                             GetStringFeaturesCallback callback)
{
    if (py_graph->graph == nullptr)
    {
        RAW_LOG_ERROR("Internal graph is not initialized");
        return 1;
    }

    std::vector<uint8_t> data;
    if (py_graph->graph->graph)
    {
        py_graph->graph->graph->GetEdgeStringFeature(
            std::span(reinterpret_cast<snark::NodeId *>(edge_src_ids), edge_size),
            std::span(reinterpret_cast<snark::NodeId *>(edge_dst_ids), edge_size),
            std::span(reinterpret_cast<snark::Type *>(edge_types), edge_size), std::span(features, features_size),
            std::span(dimensions, features_size * edge_size), data);
    }

    else
    {
        try
        {
            py_graph->graph->client->GetEdgeStringFeature(
                std::span(reinterpret_cast<snark::NodeId *>(edge_src_ids), edge_size),
                std::span(reinterpret_cast<snark::NodeId *>(edge_dst_ids), edge_size),
                std::span(reinterpret_cast<snark::Type *>(edge_types), edge_size), std::span(features, features_size),
                std::span(dimensions, features_size * edge_size), data);
        }
        catch (const std::exception &e)
        {
            RAW_LOG_ERROR("Exception while fetching node features: %s", e.what());
            return 1;
        }
    }

    callback(data.size(), data.data());
    return 0;
}

int32_t GetNeighborsInternal(PyGraph *py_graph, NodeID *in_node_ids, size_t in_node_ids_size, Type *in_edge_types,
                             size_t in_edge_types_size, uint64_t *out_neighbor_counts,
                             std::vector<NodeID> &out_neighbor_ids, std::vector<Type> &out_edge_types,
                             std::vector<float> &out_edge_weights)
{
    if (py_graph->graph == nullptr)
    {
        RAW_LOG_ERROR("Internal graph is not initialized");
        return 1;
    }

    std::fill_n(out_neighbor_counts, in_node_ids_size, 0);
    if (py_graph->graph->graph)
    {
        py_graph->graph->graph->FullNeighbor(
            std::span(reinterpret_cast<snark::NodeId *>(in_node_ids), in_node_ids_size),
            std::span(reinterpret_cast<snark::Type *>(in_edge_types), in_edge_types_size), out_neighbor_ids,
            out_edge_types, out_edge_weights, std::span(out_neighbor_counts, in_node_ids_size));
        return 0;
    }

    try
    {
        py_graph->graph->client->FullNeighbor(
            std::span(reinterpret_cast<snark::NodeId *>(in_node_ids), in_node_ids_size),
            std::span(reinterpret_cast<snark::Type *>(in_edge_types), in_edge_types_size), out_neighbor_ids,
            out_edge_types, out_edge_weights, std::span(out_neighbor_counts, in_node_ids_size));

        return 0;
    }
    catch (const std::exception &e)
    {
        RAW_LOG_ERROR("Exception while sampling neighbors: %s", e.what());
        return 1;
    }

    return 0;
}

int32_t NeighborCount(PyGraph *py_graph, NodeID *in_node_ids, size_t in_node_ids_size, Type *in_edge_types,
                      size_t in_edge_types_size, uint64_t *out_neighbor_counts)
{
    if (py_graph->graph == nullptr)
    {
        RAW_LOG_ERROR("Internal graph is not initialized");
        return 1;
    }

    if (py_graph->graph->graph)
    {
        py_graph->graph->graph->NeighborCount(
            std::span(reinterpret_cast<snark::NodeId *>(in_node_ids), in_node_ids_size),
            std::span(reinterpret_cast<snark::Type *>(in_edge_types), in_edge_types_size),
            std::span(out_neighbor_counts, in_node_ids_size));
        return 0;
    }

    try
    {
        py_graph->graph->client->NeighborCount(
            std::span(reinterpret_cast<snark::NodeId *>(in_node_ids), in_node_ids_size),
            std::span(reinterpret_cast<snark::Type *>(in_edge_types), in_edge_types_size),
            std::span(out_neighbor_counts, in_node_ids_size));
        return 0;
    }
    catch (const std::exception &e)
    {
        RAW_LOG_ERROR("Exception while fetching neighbor counts: %s", e.what());
        return 1;
    }

    return 0;
}

int32_t GetNeighbors(PyGraph *py_graph, NodeID *in_node_ids, size_t in_node_ids_size, Type *in_edge_types,
                     size_t in_edge_types_size, uint64_t *out_neighbor_counts, GetNeighborsCallback callback)
{
    std::vector<snark::NodeId> neighbor_ids;
    std::vector<snark::Type> edge_types;
    std::vector<float> edge_weights;
    std::fill_n(out_neighbor_counts, in_node_ids_size, 0);
    const auto res = GetNeighborsInternal(py_graph, in_node_ids, in_node_ids_size, in_edge_types, in_edge_types_size,
                                          out_neighbor_counts, neighbor_ids, edge_types, edge_weights);
    if (res != 0)
    {
        return res;
    }

    callback(neighbor_ids.data(), edge_weights.data(), edge_types.data(), neighbor_ids.size());
    return 0;
}

int32_t WeightedSampleNeighbor(PyGraph *py_graph, int64_t seed, NodeID *in_node_ids, size_t in_node_ids_size,
                               Type *in_edge_types, size_t in_edge_types_size, size_t count, NodeID *out_neighbor_ids,
                               Type *out_types, float *out_weights, NodeID default_node_id, float default_weight,
                               Type default_edge_type)
{
    if (py_graph->graph == nullptr)
    {
        RAW_LOG_ERROR("Internal graph is not initialized");
        return 1;
    }

    const auto out_size = count * in_node_ids_size;
    std::vector<float> total_neighbor_weights(in_node_ids_size);
    if (py_graph->graph->graph)
    {
        py_graph->graph->graph->SampleNeighbor(
            seed, std::span(reinterpret_cast<snark::NodeId *>(in_node_ids), in_node_ids_size),
            std::span(reinterpret_cast<snark::Type *>(in_edge_types), in_edge_types_size), count,
            std::span(reinterpret_cast<snark::NodeId *>(out_neighbor_ids), out_size),
            std::span(reinterpret_cast<snark::Type *>(out_types), out_size),
            std::span(reinterpret_cast<float *>(out_weights), out_size), std::span(total_neighbor_weights),
            default_node_id, default_weight, default_edge_type);

        return 0;
    }

    try
    {
        py_graph->graph->client->WeightedSampleNeighbor(
            seed, std::span(reinterpret_cast<snark::NodeId *>(in_node_ids), in_node_ids_size),
            std::span(reinterpret_cast<snark::Type *>(in_edge_types), in_edge_types_size), count,
            std::span(reinterpret_cast<snark::NodeId *>(out_neighbor_ids), out_size),
            std::span(reinterpret_cast<snark::Type *>(out_types), out_size),
            std::span(reinterpret_cast<float *>(out_weights), out_size), default_node_id, default_weight,
            default_edge_type);

        return 0;
    }
    catch (const std::exception &e)
    {
        RAW_LOG_ERROR("Exception while sampling neighbors: %s", e.what());
        return 1;
    }
}

int32_t UniformSampleNeighbor(PyGraph *py_graph, bool without_replacement, int64_t seed, NodeID *in_node_ids,
                              size_t in_node_ids_size, Type *in_edge_types, size_t in_edge_types_size, size_t count,
                              NodeID *out_neighbor_ids, Type *out_types, NodeID default_node_id, Type default_edge_type)
{
    if (py_graph->graph == nullptr)
    {
        RAW_LOG_ERROR("Internal graph is not initialized");
        return 1;
    }

    const auto out_size = count * in_node_ids_size;
    if (py_graph->graph->graph)
    {
        std::vector<uint64_t> total_neighbor_counts(in_node_ids_size);
        py_graph->graph->graph->UniformSampleNeighbor(
            without_replacement, seed, std::span(reinterpret_cast<snark::NodeId *>(in_node_ids), in_node_ids_size),
            std::span(reinterpret_cast<snark::Type *>(in_edge_types), in_edge_types_size), count,
            std::span(reinterpret_cast<snark::NodeId *>(out_neighbor_ids), out_size),
            std::span(reinterpret_cast<snark::Type *>(out_types), out_size), std::span(total_neighbor_counts),
            default_node_id, default_edge_type);

        return 0;
    }

    try
    {
        py_graph->graph->client->UniformSampleNeighbor(
            without_replacement, seed, std::span(reinterpret_cast<snark::NodeId *>(in_node_ids), in_node_ids_size),
            std::span(reinterpret_cast<snark::Type *>(in_edge_types), in_edge_types_size), count,
            std::span(reinterpret_cast<snark::NodeId *>(out_neighbor_ids), out_size),
            std::span(reinterpret_cast<snark::Type *>(out_types), out_size), default_node_id, default_edge_type);

        return 0;
    }
    catch (const std::exception &e)
    {
        RAW_LOG_ERROR("Exception while sampling neighbors: %s", e.what());
        return 1;
    }
}

// Expected length of out_node_ids buffer is (walk_length + 1) * in_node_ids_size
int32_t RandomWalk(PyGraph *py_graph, int64_t seed, float p, float q, NodeID default_node_id, NodeID *in_node_ids,
                   size_t in_node_ids_size, Type *in_edge_types, size_t in_edge_types_size, size_t walk_length,
                   NodeID *out_node_ids)
{
    if (py_graph->graph == nullptr)
    {
        RAW_LOG_ERROR("Internal graph is not initialized");
        return 1;
    }

    if (walk_length == 0)
    {
        std::copy_n(in_node_ids, in_node_ids_size, out_node_ids);
        return 0;
    }

    const bool distributed_setting = py_graph->graph->client != nullptr;
    snark::Xoroshiro128PlusGenerator gen(seed);

    // We use single precision everywhere, so in case we'll need a better accuracy we can replace float to double
    // everywhere in this function.
    boost::random::uniform_real_distribution<float> d(0, 1);
    std::fill_n(out_node_ids, (walk_length + 1) * in_node_ids_size, default_node_id);
    for (size_t index = 0; index < in_node_ids_size; ++index)
    {
        out_node_ids[index * (walk_length + 1)] = in_node_ids[index];
    }

    // Naming convention follows node2vec paper: t_nbs - neighbors of a parent node. We skip v_nbs
    // for current node neighbors, for performance reasons: use less memory and deterministic traversal.
    // Flat_hash_map iteration is random with the same data, but with different processes.
    std::vector<absl::flat_hash_map<NodeID, float>> t_nbs(in_node_ids_size);

    // Containers for current and pass nodes in the walk.
    std::vector<NodeID> curr_nodes(in_node_ids, in_node_ids + in_node_ids_size);
    std::vector<NodeID> past_nodes(in_node_ids, in_node_ids + in_node_ids_size);
    std::vector<uint64_t> curr_counts(in_node_ids_size);

    // Unnormalized transition probabilities to node x if the parent is t and current node is v:
    //           / 1/p if d_{tx} = 0
    // P[t, x] = |  1  if d_{tx} = 1
    //           \ 1/q if d_{tx} = 2
    std::vector<std::pair<NodeID, float>> transition_node_prob;
    std::vector<NodeID> transition_nodes;
    std::vector<float> transition_probs;
    auto curr_out_nodes = out_node_ids;
    std::vector<NodeID> neighbors;
    std::vector<float> weights;
    std::vector<Type> types;
    for (size_t curr_step = 0; curr_step < walk_length; ++curr_step)
    {
        neighbors.resize(0);
        weights.resize(0);
        types.resize(0);
        GetNeighborsInternal(py_graph, curr_nodes.data(), curr_nodes.size(), in_edge_types, in_edge_types_size,
                             curr_counts.data(), neighbors, types, weights);
        size_t nb_offset = 0;
        for (size_t index = 0; index < in_node_ids_size; ++index)
        {
            // Look up parent transitional probability via map to find edge weight.
            auto parent = t_nbs[index].find(curr_nodes[index]);
            if (parent != std::end(t_nbs[index]))
            {
                transition_node_prob.emplace_back(past_nodes[index], parent->second / p);
            }

            for (size_t nb_index = 0; nb_index < curr_counts[index]; ++nb_index, ++nb_offset)
            {
                const auto node = neighbors[nb_offset];
                const auto weight = weights[nb_offset];
                // Skip parent node, because it was added above.
                if (node == past_nodes[index])
                {
                    continue;
                }

                // Found current neighbor in parents neighbors.
                else if (t_nbs[index].find(node) != std::end(t_nbs[index]))
                {
                    transition_node_prob.emplace_back(node, weight);
                }
                // Explore current node neighborhood.
                else
                {
                    transition_node_prob.emplace_back(node, weight / q);
                }
            }

            if (transition_node_prob.empty())
            {
                past_nodes[index] = curr_nodes[index];
                curr_nodes[index] = default_node_id;
                curr_out_nodes[index * (walk_length + 1) + curr_step + 1] = default_node_id;
                continue;
            }

            // We need to sort the final neighbors list to get deterministic results, because in distributed setting
            // shards can return results in random order.
            if (distributed_setting)
            {
                std::sort(std::begin(transition_node_prob), std::end(transition_node_prob));
            }
            transition_nodes.reserve(transition_node_prob.size());
            transition_probs.reserve(transition_node_prob.size());

            // Unfortunatelly we can't use distributions from the standard library here:
            // 1. piecewise_constant_distribution uses real numbers to return as items,
            // which might be not accurate for int64_t.
            // 2. discrete_distribution is using integer weights, but we need floats here.
            float total_weight = 0;
            for (const auto &p : transition_node_prob)
            {
                transition_nodes.emplace_back(p.first);
                total_weight += p.second;
                transition_probs.emplace_back(total_weight);
            }

            const auto random_number = total_weight * d(gen);
            auto offset = std::lower_bound(std::begin(transition_probs), std::end(transition_probs), random_number);
            NodeID next_node = transition_nodes[offset - std::begin(transition_probs)];
            past_nodes[index] = curr_nodes[index];
            curr_nodes[index] = next_node;
            curr_out_nodes[index * (walk_length + 1) + curr_step + 1] = next_node;
            transition_nodes.resize(0);
            transition_probs.resize(0);
            transition_node_prob.resize(0);
            t_nbs[index].clear();
            t_nbs[index].reserve(curr_counts[index]);

            // Return global index back to replace parent nodes.
            nb_offset -= curr_counts[index];
            for (size_t nb_index = 0; nb_index < curr_counts[index]; ++nb_index, ++nb_offset)
            {
                t_nbs[index].emplace(neighbors[nb_offset], weights[nb_offset]);
            }
        }
    }
    return 0;
}

namespace
{

// Auxilarry data structure to group all variables in one place.
typedef struct
{
    absl::flat_hash_map<NodeID, size_t> lookup;
    std::vector<NodeID> node_ids;
    std::vector<size_t> nb_index;
    std::vector<uint64_t> counts;
} NB_Count_Cache;

// Cache neighbor counts for each node in the input list.
void lookup_neighbor_counts(PyGraph *py_graph, NB_Count_Cache &cache, const std::vector<NodeID> &neighbors,
                            Type *in_edge_types, size_t in_edge_types_size, std::vector<uint64_t> &neighbor_counts)
{
    assert(cache.node_ids.empty());
    assert(cache.counts.empty());
    assert(cache.nb_index.empty());

    for (size_t nb_index = 0; nb_index < neighbors.size(); ++nb_index)
    {
        auto it = cache.lookup.find(neighbors[nb_index]);
        if (it != cache.lookup.end())
        {
            neighbor_counts[nb_index] = it->second;
        }
        else
        {
            cache.node_ids.emplace_back(neighbors[nb_index]);
            cache.nb_index.emplace_back(nb_index);
        }
    }

    if (cache.node_ids.empty())
    {
        return;
    }

    const size_t nodes_size = cache.node_ids.size();
    cache.counts.resize(nodes_size);
    NeighborCount(py_graph, cache.node_ids.data(), nodes_size, in_edge_types, in_edge_types_size, cache.counts.data());
    for (size_t i = 0; i < nodes_size; ++i)
    {
        cache.lookup[cache.node_ids[i]] = cache.counts[i];
        neighbor_counts[cache.nb_index[i]] = cache.counts[i];
    }

    cache.node_ids.clear();
    cache.counts.clear();
    cache.nb_index.clear();
}
} // namespace

// Implementation of PPR-go is based on https://github.com/TUM-DAML/pprgo_pytorch/blob/master/pprgo/ppr.py
int32_t PPRSampleNeighbor(PyGraph *py_graph, NodeID *in_node_ids, size_t int_node_ids_size, Type *in_edge_types,
                          size_t in_edge_types_size, const size_t count, const float alpha, const float eps,
                          const NodeID default_node_id, const float default_weight, NodeID *out_neighbor_ids,
                          float *out_weights)
{
    if (py_graph->graph == nullptr)
    {
        RAW_LOG_ERROR("Internal graph is not initialized");
        return 1;
    }

    const float alpha_eps = alpha * eps;
    std::vector<NodeID> q;

    std::vector<NodeID> neighbors;
    std::vector<float> weights;
    std::vector<Type> types;

    absl::flat_hash_map<NodeID, float> p;
    absl::flat_hash_map<NodeID, float> r;
    using WN = std::pair<float, NodeID>;
    std::priority_queue<WN, std::vector<WN>, std::greater<WN>> pq;
    std::vector<uint64_t> neighbor_counts;
    NB_Count_Cache nb_cache;

    for (size_t node_index = 0; node_index < int_node_ids_size; ++node_index)
    {
        p.clear();
        r.clear();
        const auto inode = in_node_ids[node_index];
        r[inode] = alpha;
        q.emplace_back(inode);
        while (!q.empty())
        {
            auto unode = q.back();
            q.pop_back();
            float res = 0;
            auto r_unode = r.find(unode);
            if (r_unode != std::end(r))
            {
                res = r_unode->second;
            }
            p[unode] += res;
            r[unode] = 0;

            neighbors.resize(0);
            weights.resize(0);
            types.resize(0);
            uint64_t curr_count = 0;
            GetNeighborsInternal(py_graph, &unode, 1, in_edge_types, in_edge_types_size, &curr_count, neighbors, types,
                                 weights);
            neighbor_counts.resize(neighbors.size());
            lookup_neighbor_counts(py_graph, nb_cache, neighbors, in_edge_types, in_edge_types_size, neighbor_counts);
            const float _val = (1 - alpha) * res / neighbors.size();
            for (size_t nb_index = 0; nb_index < neighbors.size(); ++nb_index)
            {
                auto vnode = neighbors[nb_index];
                r[vnode] += _val;

                float res_vnode = 0;
                auto r_vnode = r.find(vnode);
                if (r_vnode != std::end(r))
                {
                    res_vnode = r_vnode->second;
                }

                if (res_vnode >= alpha_eps * neighbor_counts[nb_index])
                {
                    auto f = std::find(std::begin(q), std::end(q), vnode);
                    if (f == std::end(q))
                    {
                        q.emplace_back(vnode);
                    }
                }
            }
        }

        for (auto kv : p)
        {
            if (pq.size() < count)
            {
                pq.emplace(kv.second, kv.first);
            }
            else
            {
                const auto &top = pq.top();
                if (top.first > kv.second)
                {
                    continue;
                }
                pq.pop();
                pq.emplace(kv.second, kv.first);
            }
        }

        for (size_t ppr_count = 0; ppr_count < count; ++ppr_count)
        {
            const auto out_index = node_index * count + ppr_count;
            if (pq.empty())
            {
                out_neighbor_ids[out_index] = default_node_id;
                out_weights[out_index] = default_weight;
                continue;
            }

            auto q = pq.top();
            out_neighbor_ids[out_index] = q.second;
            out_weights[out_index] = q.first;
            pq.pop();
        }

        assert(pq.empty());
    }

    return 0;
}

int32_t ResetSampler(PySampler *py_sampler)
{
    py_sampler->sampler.reset();
    return 0;
}

int32_t ResetGraph(PyGraph *py_graph)
{
    py_graph->graph.reset();
    return 0;
}

int32_t HDFSMoveMeta(const char *filename_src, const char *filename_dst, const char *config_path)
{
    auto data = read_hdfs<char>(filename_src, config_path);
    if (data.size() > 100000) // 100kB defined as max size of meta
    {
        RAW_LOG_ERROR("HDFSReadMeta meta.txt too large, %li > 100kB!", data.size());
        return 1;
    }

    auto file = fopen(filename_dst, "w");
    if (file == nullptr)
    {
        RAW_LOG_ERROR("Failed to open meta.txt for writing at '%s'!", filename_dst);
        return 1;
    }
    fwrite(data.data(), sizeof(char), data.size(), file);
    fclose(file);

    return 0;
}

} // namespace python
} // namespace deep_graph
