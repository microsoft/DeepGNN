// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "client.h"

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cstdio>
#include <exception>
#include <functional>
#include <future>
#include <limits>
#include <numeric>
#include <random>
#include <thread>
#include <type_traits>

#include "src/cc/lib/distributed/call_data.h"
#include "src/cc/lib/graph/xoroshiro.h"

// Use raw log to avoid possible initialization conflicts with glog from other libraries.
#include <glog/logging.h>
#include <glog/raw_logging.h>

#include "boost/random/binomial_distribution.hpp"
#include "boost/random/uniform_int_distribution.hpp"
#include "boost/random/uniform_real_distribution.hpp"
#include <grpcpp/client_context.h>
#include <grpcpp/create_channel.h>

namespace
{
using grpc::ClientContext;
using grpc::Status;
// Client queue item with information about request.
struct AsyncClientCall
{
    ClientContext context;

    // Status is shared accross all types of requests
    Status status;

    // Callback to execute after recieving response from the server.
    std::function<void()> callback;

    // Promise set by the client after executing callback.
    std::promise<void> promise;
};

// Index to look up feature coordinates to return them in sorted order.
// shard, index offset, index count, value offset, value count
using SparseFeatureIndex = std::tuple<size_t, int, int, int, int>;

void WaitForFutures(std::vector<std::future<void>> &futures)
{
    for (auto &f : futures)
    {
        f.get();
    }
}
void ExtractFeatures(const std::vector<std::vector<SparseFeatureIndex>> &response_index,
                     const std::vector<snark::SparseFeaturesReply> &replies, std::span<int64_t> out_dimensions,
                     std::vector<std::vector<int64_t>> &out_indices, std::vector<std::vector<uint8_t>> &out_values,
                     size_t node_count)
{
    for (size_t feature_index = 0; feature_index < out_dimensions.size(); ++feature_index)
    {
        for (size_t node_index = 0; node_index < node_count; ++node_index)
        {
            size_t shard;
            int index_start, index_count, value_start, value_count;
            std::tie(shard, index_start, index_count, value_start, value_count) =
                response_index[node_index][feature_index];
            if (replies[shard].indices().empty())
            {
                continue;
            }
            std::copy_n(std::begin(replies[shard].indices()) + index_start, index_count,
                        std::back_inserter(out_indices[feature_index]));
            std::copy_n(std::begin(replies[shard].values()) + value_start, value_count,
                        std::back_inserter(out_values[feature_index]));
        }
    }
}

void ExtractStringFeatures(const std::vector<std::pair<size_t, size_t>> &response_index,
                           const std::vector<snark::StringFeaturesReply> &replies, std::span<int64_t> dimensions,
                           std::vector<uint8_t> &out_values)
{
    const auto total_size = std::accumulate(std::begin(dimensions), std::end(dimensions), int64_t(0));
    out_values.reserve(total_size);
    for (size_t feature_index = 0; feature_index < dimensions.size(); ++feature_index)
    {
        const auto &index = response_index[feature_index];
        std::copy_n(std::begin(replies[index.first].values()) + index.second, dimensions[feature_index],
                    std::back_inserter(out_values));
    }
}

template <typename SparseRequest, typename NextCompletionQueue>
void GetSparseFeature(const SparseRequest &request,
                      std::vector<std::unique_ptr<snark::GraphEngine::Stub>> &engine_stubs, size_t input_size,
                      size_t feature_count, std::span<int64_t> out_dimensions,
                      std::vector<std::vector<int64_t>> &out_indices, std::vector<std::vector<uint8_t>> &out_values,
                      NextCompletionQueue next_completion_queue)
{
    std::vector<std::future<void>> futures;
    futures.reserve(engine_stubs.size());
    std::vector<snark::SparseFeaturesReply> replies(engine_stubs.size());
    std::vector<std::vector<SparseFeatureIndex>> response_index(input_size,
                                                                std::vector<SparseFeatureIndex>(feature_count));

    for (size_t shard = 0; shard < engine_stubs.size(); ++shard)
    {
        auto *call = new AsyncClientCall();
        std::unique_ptr<grpc::ClientAsyncResponseReader<snark::SparseFeaturesReply>> response_reader;
        if constexpr (std::is_same<SparseRequest, snark::NodeSparseFeaturesRequest>::value)
        {
            response_reader = engine_stubs[shard]->PrepareAsyncGetNodeSparseFeatures(&call->context, request,
                                                                                     next_completion_queue());
        }
        else if constexpr (std::is_same<SparseRequest, snark::EdgeSparseFeaturesRequest>::value)
        {
            response_reader = engine_stubs[shard]->PrepareAsyncGetEdgeSparseFeatures(&call->context, request,
                                                                                     next_completion_queue());
        }
        else
        {
            throw std::runtime_error("Unknown request type for GetSparseFeature");
        }

        call->callback = [&reply = replies[shard], &response_index, shard, out_dimensions]() {
            if (reply.indices().empty())
            {
                return;
            }
            int64_t node_offset = 0;
            int64_t value_offset = 0;
            int64_t node_cummulative_count = 0;
            for (int64_t feature_index = 0; feature_index < reply.dimensions().size(); ++feature_index)
            {
                const auto feature_dim = reply.dimensions(feature_index) + 1;
                if (feature_dim == 1)
                {
                    continue;
                }

                if (out_dimensions[feature_index] != 0 && reply.dimensions(feature_index) != 0 &&
                    out_dimensions[feature_index] != reply.dimensions(feature_index))
                {
                    auto feature_str = std::to_string(feature_index);
                    auto client_dimension_str = std::to_string(out_dimensions[feature_index]);
                    auto server_dimension_str = std::to_string(reply.dimensions(feature_index));
                    RAW_LOG_FATAL("Dimensions do not match for sparse feature %s. %s != %s", feature_str.c_str(),
                                  client_dimension_str.c_str(), server_dimension_str.c_str());
                }

                out_dimensions[feature_index] = reply.dimensions(feature_index);
                const size_t value_increment =
                    (reply.values_counts(feature_index) * feature_dim) / reply.indices_counts(feature_index);
                node_cummulative_count += reply.indices_counts(feature_index);
                for (; node_offset < node_cummulative_count;
                     node_offset += feature_dim, value_offset += value_increment)
                {
                    const auto item_index = reply.indices(node_offset);
                    int64_t count = std::get<2>(response_index[item_index][feature_index]);
                    if (count == 0)
                    {
                        std::get<0>(response_index[item_index][feature_index]) = shard;
                        std::get<1>(response_index[item_index][feature_index]) = node_offset;
                        std::get<3>(response_index[item_index][feature_index]) = value_offset;
                    }
                    std::get<2>(response_index[item_index][feature_index]) += feature_dim;
                    std::get<4>(response_index[item_index][feature_index]) += value_increment;
                }
            }
        };

        // Order is important: call variable can be deleted before futures can be
        // obtained.
        futures.emplace_back(call->promise.get_future());
        response_reader->StartCall();
        response_reader->Finish(&replies[shard], &call->status, static_cast<void *>(call));
    }

    WaitForFutures(futures);
    ExtractFeatures(response_index, replies, out_dimensions, out_indices, out_values, input_size);
}

template <typename SparseRequest, typename NextCompletionQueue>
void GetStringFeature(const SparseRequest &request,
                      std::vector<std::unique_ptr<snark::GraphEngine::Stub>> &engine_stubs, size_t input_size,
                      size_t feature_count, std::span<int64_t> out_dimensions, std::vector<uint8_t> &out_values,
                      NextCompletionQueue next_completion_queue)
{
    std::vector<std::future<void>> futures;
    futures.reserve(engine_stubs.size());
    std::vector<snark::StringFeaturesReply> replies(engine_stubs.size());
    std::vector<std::pair<size_t, size_t>> response_index(input_size * feature_count);

    for (size_t shard = 0; shard < engine_stubs.size(); ++shard)
    {
        auto *call = new AsyncClientCall();
        std::unique_ptr<grpc::ClientAsyncResponseReader<snark::StringFeaturesReply>> response_reader;
        if constexpr (std::is_same<SparseRequest, snark::NodeSparseFeaturesRequest>::value)
        {
            response_reader = engine_stubs[shard]->PrepareAsyncGetNodeStringFeatures(&call->context, request,
                                                                                     next_completion_queue());
        }
        else if constexpr (std::is_same<SparseRequest, snark::EdgeSparseFeaturesRequest>::value)
        {
            response_reader = engine_stubs[shard]->PrepareAsyncGetEdgeStringFeatures(&call->context, request,
                                                                                     next_completion_queue());
        }
        else
        {
            throw std::runtime_error("Unknown request type for GetStringFeature");
        }

        call->callback = [&reply = replies[shard], &response_index, shard, out_dimensions]() {
            if (reply.values().empty())
            {
                return;
            }
            int64_t value_offset = 0;
            for (int64_t feature_index = 0; feature_index < reply.dimensions().size(); ++feature_index)
            {
                const auto dim = reply.dimensions(feature_index);
                if (dim == 0)
                {
                    continue;
                }

                // it is ok to not to synchronize here, because feature data is the same across shards.
                response_index[feature_index] = {shard, value_offset};
                out_dimensions[feature_index] = dim;
                value_offset += dim;
            }
        };

        // Order is important: call variable can be deleted before futures can be
        // obtained.
        futures.emplace_back(call->promise.get_future());
        response_reader->StartCall();
        response_reader->Finish(&replies[shard], &call->status, static_cast<void *>(call));
    }

    WaitForFutures(futures);
    ExtractStringFeatures(response_index, replies, out_dimensions, out_values);
}

} // namespace

namespace snark
{

using grpc::ClientContext;
using grpc::ClientReader;
using grpc::ClientReaderWriter;
using grpc::ClientWriter;
using grpc::Status;

GRPCClient::GRPCClient(std::vector<std::shared_ptr<grpc::Channel>> channels, uint32_t num_threads,
                       uint32_t num_threads_per_cq)
{
    num_threads = std::max(uint32_t(1), num_threads);
    num_threads_per_cq = std::max(uint32_t(1), num_threads_per_cq);
    uint32_t num_cqs = (num_threads + num_threads_per_cq - 1) / num_threads_per_cq;
    m_completion_queue = std::vector<grpc::CompletionQueue>(num_cqs);
    for (auto c : channels)
    {
        m_engine_stubs.emplace_back(snark::GraphEngine::NewStub(c));
        m_sampler_stubs.emplace_back(snark::GraphSampler::NewStub(c));
    }

    for (uint32_t i = 0; i < num_threads; ++i)
    {
        m_reply_threads.emplace_back(AsyncCompleteRpc(i % m_completion_queue.size()));
    }
}

std::function<void()> GRPCClient::AsyncCompleteRpc(size_t index)
{
    return [&queue = m_completion_queue[index]]() {
        void *got_tag;
        bool ok = false;

        while (queue.Next(&got_tag, &ok))
        {
            GPR_ASSERT(ok);
            auto call = static_cast<AsyncClientCall *>(got_tag);

            if (call->status.ok())
            {
                try
                {
                    call->callback();
                    call->promise.set_value();
                }
                catch (const std::exception &e)
                {
                    RAW_LOG_ERROR("Client failed to process request. Exception: %s", e.what());
                    call->promise.set_exception(std::current_exception());
                }
            }
            else
            {
                RAW_LOG_ERROR("Request failed, code: %d. Message: %s", call->status.error_code(),
                              call->status.error_message().c_str());
                try
                {
                    throw std::runtime_error(std::string("Request failed. Message: ") + call->status.error_message());
                }
                catch (std::exception &e)
                {
                    call->promise.set_exception(std::current_exception());
                }
            }

            // Once we're complete, deallocate the call object.
            delete call;
        }
    };
}

void GRPCClient::GetNodeType(std::span<const NodeId> node_ids, std::span<Type> output, Type default_type)
{
    assert(node_ids.size() == output.size());

    NodeTypesRequest request;
    const auto node_len = node_ids.size();
    *request.mutable_node_ids() = {std::begin(node_ids), std::end(node_ids)};
    std::vector<std::future<void>> futures;
    futures.reserve(m_engine_stubs.size());
    std::vector<NodeTypesReply> replies(m_engine_stubs.size());

    // Vector<bool> is not thread safe for our use case, because it's storage is not contiguous
    auto found = std::make_unique<bool[]>(node_len);
    for (size_t shard = 0; shard < m_engine_stubs.size(); ++shard)
    {
        auto *call = new AsyncClientCall();

        auto response_reader =
            m_engine_stubs[shard]->PrepareAsyncGetNodeTypes(&call->context, request, NextCompletionQueue());

        call->callback = [&reply = replies[shard], output, &found]() {
            if (reply.offsets().empty())
            {
                return;
            }

            auto curr_type_reply = std::begin(reply.types());
            for (auto index : reply.offsets())
            {
                output[index] = *curr_type_reply;
                found[index] = true;
                ++curr_type_reply;
            }
        };

        // Order is important: call variable can be deleted before futures can be
        // obtained.
        futures.emplace_back(call->promise.get_future());
        response_reader->StartCall();
        response_reader->Finish(&replies[shard], &call->status, static_cast<void *>(call));
    }

    WaitForFutures(futures);
    for (size_t i = 0; i < node_len; ++i)
    {
        if (!found[i])
        {
            output[i] = default_type;
        }
    }
}

void GRPCClient::GetNodeFeature(std::span<const NodeId> node_ids, std::span<FeatureMeta> features,
                                std::span<uint8_t> output)
{
    assert(std::accumulate(std::begin(features), std::end(features), size_t(0),
                           [](size_t val, const auto &f) { return val + f.second; }) *
               node_ids.size() ==
           output.size());

    NodeFeaturesRequest request;
    const auto node_len = node_ids.size();
    *request.mutable_node_ids() = {std::begin(node_ids), std::end(node_ids)};
    for (const auto &feature : features)
    {
        auto wire_feature = request.add_features();
        wire_feature->set_id(feature.first);
        wire_feature->set_size(feature.second);
    }
    const size_t fv_size = output.size() / node_len;
    std::vector<std::future<void>> futures;
    futures.reserve(m_engine_stubs.size());
    std::vector<NodeFeaturesReply> replies(m_engine_stubs.size());

    // Vector<bool> is not thread safe for our use case, because it's storage is not contiguous
    auto found = std::make_unique<bool[]>(node_len);
    for (size_t shard = 0; shard < m_engine_stubs.size(); ++shard)
    {
        auto *call = new AsyncClientCall();

        auto response_reader =
            m_engine_stubs[shard]->PrepareAsyncGetNodeFeatures(&call->context, request, NextCompletionQueue());

        call->callback = [&reply = replies[shard], output, &found, fv_size]() {
            if (reply.offsets().empty())
            {
                return;
            }

            auto curr_feature_out = std::begin(output);
            // Use c_str since string iterators can process wide charachters on windows.
            auto curr_feature_reply = reply.feature_values().c_str();
            for (auto index : reply.offsets())
            {
                std::copy(curr_feature_reply, curr_feature_reply + fv_size, curr_feature_out + fv_size * index);
                curr_feature_reply += fv_size;
                found[index] = true;
            }
        };

        // Order is important: call variable can be deleted before futures can be
        // obtained.
        futures.emplace_back(call->promise.get_future());
        response_reader->StartCall();
        response_reader->Finish(&replies[shard], &call->status, static_cast<void *>(call));
    }

    WaitForFutures(futures);
    auto values = std::begin(output);
    for (size_t i = 0; i < node_len; ++i)
    {
        if (found[i])
        {
            values += fv_size;
        }
        else
        {
            values = std::fill_n(values, fv_size, 0);
        }
    }
}

void GRPCClient::GetEdgeFeature(std::span<const NodeId> edge_src_ids, std::span<const NodeId> edge_dst_ids,
                                std::span<const Type> edge_types, std::span<FeatureMeta> features,
                                std::span<uint8_t> output)
{
    const auto len = edge_types.size();
    assert(std::accumulate(std::begin(features), std::end(features), size_t(0),
                           [](size_t val, const auto &f) { return val + f.second; }) *
               len ==
           output.size());
    assert(len == edge_src_ids.size());
    assert(len == edge_dst_ids.size());
    assert(output.size() % len == 0);

    EdgeFeaturesRequest request;
    *request.mutable_node_ids() = {std::begin(edge_src_ids), std::end(edge_src_ids)};
    request.mutable_node_ids()->Add(std::begin(edge_dst_ids), std::end(edge_dst_ids));
    request.mutable_types()->Add(std::begin(edge_types), std::end(edge_types));
    for (const auto &feature : features)
    {
        auto wire_feature = request.add_features();
        wire_feature->set_id(feature.first);
        wire_feature->set_size(feature.second);
    }

    const size_t fv_size = output.size() / len;
    std::vector<std::future<void>> futures;
    futures.reserve(m_engine_stubs.size());
    std::vector<EdgeFeaturesReply> replies(m_engine_stubs.size());

    // Vector<bool> is not thread safe for our use case, because it's storage is not contiguous
    auto found = std::make_unique<bool[]>(len);
    for (size_t shard = 0; shard < m_engine_stubs.size(); ++shard)
    {
        auto *call = new AsyncClientCall();

        auto response_reader =
            m_engine_stubs[shard]->PrepareAsyncGetEdgeFeatures(&call->context, request, NextCompletionQueue());

        call->callback = [&reply = replies[shard], output, fv_size, &found]() {
            if (reply.offsets().empty())
            {
                return;
            }

            auto curr_feature_out = std::begin(output);
            // Use c_str since string iterators can process wide charachters on windows.
            auto curr_feature_reply = reply.feature_values().c_str();
            for (auto index : reply.offsets())
            {
                std::copy(curr_feature_reply, curr_feature_reply + fv_size, curr_feature_out + fv_size * index);
                curr_feature_reply += fv_size;
                found[index] = true;
            }
        };

        // Order is important: call variable can be deleted before futures can be
        // obtained.
        futures.emplace_back(call->promise.get_future());
        response_reader->StartCall();
        response_reader->Finish(&replies[shard], &call->status, static_cast<void *>(call));
    }

    WaitForFutures(futures);
    auto values = std::begin(output);
    for (size_t i = 0; i < len; ++i)
    {
        if (found[i])
        {
            values += fv_size;
        }
        else
        {
            values = std::fill_n(values, fv_size, 0);
        }
    }
}

void GRPCClient::GetNodeSparseFeature(std::span<const NodeId> node_ids, std::span<const FeatureId> features,
                                      std::span<int64_t> out_dimensions, std::vector<std::vector<int64_t>> &out_indices,
                                      std::vector<std::vector<uint8_t>> &out_values)
{
    assert(out_indices.size() == features.size());
    assert(out_dimensions.size() == features.size());

    // Fill out_dimensions in case nodes don't have some features.
    std::fill(std::begin(out_dimensions), std::end(out_dimensions), 0);
    NodeSparseFeaturesRequest request;
    *request.mutable_node_ids() = {std::begin(node_ids), std::end(node_ids)};
    *request.mutable_feature_ids() = {std::begin(features), std::end(features)};

    GetSparseFeature(request, m_engine_stubs, node_ids.size(), features.size(), out_dimensions, out_indices, out_values,
                     std::bind(&GRPCClient::NextCompletionQueue, this));
}

void GRPCClient::GetEdgeSparseFeature(std::span<const NodeId> edge_src_ids, std::span<const NodeId> edge_dst_ids,
                                      std::span<const Type> edge_types, std::span<const FeatureId> features,
                                      std::span<int64_t> out_dimensions, std::vector<std::vector<int64_t>> &out_indices,
                                      std::vector<std::vector<uint8_t>> &out_values)
{
    const auto len = edge_types.size();
    assert(len == edge_src_ids.size());
    assert(len == edge_dst_ids.size());

    EdgeSparseFeaturesRequest request;
    *request.mutable_node_ids() = {std::begin(edge_src_ids), std::end(edge_src_ids)};
    request.mutable_node_ids()->Add(std::begin(edge_dst_ids), std::end(edge_dst_ids));
    request.mutable_types()->Add(std::begin(edge_types), std::end(edge_types));
    *request.mutable_feature_ids() = {std::begin(features), std::end(features)};

    GetSparseFeature(request, m_engine_stubs, len, features.size(), out_dimensions, out_indices, out_values,
                     std::bind(&GRPCClient::NextCompletionQueue, this));
}

void GRPCClient::GetNodeStringFeature(std::span<const NodeId> node_ids, std::span<const FeatureId> features,
                                      std::span<int64_t> out_dimensions, std::vector<uint8_t> &out_values)
{
    NodeSparseFeaturesRequest request;
    *request.mutable_node_ids() = {std::begin(node_ids), std::end(node_ids)};
    *request.mutable_feature_ids() = {std::begin(features), std::end(features)};
    GetStringFeature(request, m_engine_stubs, node_ids.size(), features.size(), out_dimensions, out_values,
                     std::bind(&GRPCClient::NextCompletionQueue, this));
}

void GRPCClient::GetEdgeStringFeature(std::span<const NodeId> edge_src_ids, std::span<const NodeId> edge_dst_ids,
                                      std::span<const Type> edge_types, std::span<const FeatureId> features,
                                      std::span<int64_t> out_dimensions, std::vector<uint8_t> &out_values)
{
    const auto len = edge_types.size();
    assert(len == edge_src_ids.size());
    assert(len == edge_dst_ids.size());

    EdgeSparseFeaturesRequest request;
    *request.mutable_node_ids() = {std::begin(edge_src_ids), std::end(edge_src_ids)};
    request.mutable_node_ids()->Add(std::begin(edge_dst_ids), std::end(edge_dst_ids));
    request.mutable_types()->Add(std::begin(edge_types), std::end(edge_types));
    *request.mutable_feature_ids() = {std::begin(features), std::end(features)};

    GetStringFeature(request, m_engine_stubs, len, features.size(), out_dimensions, out_values,
                     std::bind(&GRPCClient::NextCompletionQueue, this));
}

void GRPCClient::NeighborCount(std::span<const NodeId> node_ids, std::span<const Type> edge_types,
                               std::span<uint64_t> output_neighbor_counts)
{
    GetNeighborsRequest request;

    *request.mutable_node_ids() = {std::begin(node_ids), std::end(node_ids)};
    *request.mutable_edge_types() = {std::begin(edge_types), std::end(edge_types)};

    std::vector<std::future<void>> futures;
    std::vector<GetNeighborCountsReply> replies(std::size(m_engine_stubs));
    std::atomic<size_t> responses_left{std::size(m_engine_stubs)};

    size_t len = node_ids.size();
    std::fill_n(std::begin(output_neighbor_counts), len, 0);

    for (size_t shard = 0; shard < m_engine_stubs.size(); ++shard)
    {
        auto *call = new AsyncClientCall();
        auto response_reader =
            m_engine_stubs[shard]->PrepareAsyncGetNeighborCounts(&call->context, request, NextCompletionQueue());

        call->callback = [&responses_left, &replies, &output_neighbor_counts]() {
            // Skip processing until all responses arrived. All responses are stored in the `replies` variable,
            // so we can safely return.
            if (responses_left.fetch_sub(1) > 1)
            {
                return;
            }

            for (size_t reply_index = 0; reply_index < std::size(replies); ++reply_index)
            {
                const auto &reply = replies[reply_index];
                auto output_len = output_neighbor_counts.size();
                auto reply_len = reply.neighbor_counts().size();

                // Mismatch in lengths of output and reply vectors
                std::transform(std::begin(reply.neighbor_counts()),
                               std::begin(reply.neighbor_counts()) + std::min(output_len, size_t(reply_len)),
                               std::begin(output_neighbor_counts), std::begin(output_neighbor_counts),
                               std::plus<uint64_t>());
            }
        };

        futures.emplace_back(call->promise.get_future());
        response_reader->StartCall();
        response_reader->Finish(&replies[shard], &call->status, static_cast<void *>(call));
    }
    WaitForFutures(futures);
}

void GRPCClient::FullNeighbor(std::span<const NodeId> node_ids, std::span<const Type> edge_types,
                              std::vector<NodeId> &output_nodes, std::vector<Type> &output_types,
                              std::vector<float> &output_weights, std::span<uint64_t> output_neighbor_counts)
{
    GetNeighborsRequest request;

    *request.mutable_node_ids() = {std::begin(node_ids), std::end(node_ids)};
    *request.mutable_edge_types() = {std::begin(edge_types), std::end(edge_types)};
    std::vector<std::future<void>> futures;
    std::vector<GetNeighborsReply> replies(std::size(m_engine_stubs));
    std::vector<size_t> reply_offsets(std::size(m_engine_stubs));

    // Algorithm is to wait until all responses arive and then merge them in
    // the last callback.
    std::atomic<size_t> responses_left{std::size(m_engine_stubs)};

    for (size_t shard = 0; shard < m_engine_stubs.size(); ++shard)
    {
        auto *call = new AsyncClientCall();

        auto response_reader =
            m_engine_stubs[shard]->PrepareAsyncGetNeighbors(&call->context, request, NextCompletionQueue());

        call->callback = [&responses_left, &replies, &output_nodes, &output_types, &output_weights,
                          &output_neighbor_counts, &reply_offsets]() {
            // Skip processing until all responses arrived. All responses are stored in the `replies` variable,
            // so we can safely return.
            if (responses_left.fetch_sub(1) > 1)
            {
                return;
            }
            for (size_t curr_node = 0; curr_node < std::size(output_neighbor_counts); ++curr_node)
            {
                for (size_t reply_index = 0; reply_index < std::size(replies); ++reply_index)
                {
                    const auto &reply = replies[reply_index];
                    if (size_t(reply.neighbor_counts().size()) <= curr_node)
                    {
                        auto expected = std::to_string(output_neighbor_counts.size());
                        auto received = std::to_string(reply.neighbor_counts().size());
                        // In case of a short reply, we can skip processing. Log error if it happens.
                        RAW_LOG_ERROR(
                            "Received short list of neighbor counts: %s. Expected: %s. Assuming no neighbors.",
                            received.c_str(), expected.c_str());
                        continue;
                    }

                    const auto count = reply.neighbor_counts(curr_node);
                    if (count == 0)
                    {
                        continue;
                    }

                    output_neighbor_counts[curr_node] += count;
                    const auto offset = reply_offsets[reply_index];
                    auto node_ids_start = reply.node_ids().begin() + offset;
                    output_nodes.insert(std::end(output_nodes), node_ids_start, node_ids_start + count);

                    auto edge_weights_start = reply.edge_weights().begin() + offset;
                    output_weights.insert(std::end(output_weights), edge_weights_start, edge_weights_start + count);
                    auto edge_types_start = reply.edge_types().begin() + offset;
                    output_types.insert(std::end(output_types), edge_types_start, edge_types_start + count);
                    reply_offsets[reply_index] += count;
                }
            }
        };

        futures.emplace_back(call->promise.get_future());
        response_reader->StartCall();
        response_reader->Finish(&replies[shard], &call->status, static_cast<void *>(call));
    }
    WaitForFutures(futures);
}

void GRPCClient::WeightedSampleNeighbor(int64_t seed, std::span<const NodeId> node_ids,
                                        std::span<const Type> edge_types, size_t count,
                                        std::span<NodeId> output_neighbors, std::span<Type> output_types,
                                        std::span<float> output_weights, NodeId default_node_id, float default_weight,
                                        Type default_edge_type)
{
    snark::Xoroshiro128PlusGenerator engine(seed);
    boost::random::uniform_int_distribution<int64_t> subseed(std::numeric_limits<int64_t>::min(),
                                                             std::numeric_limits<int64_t>::max());

    WeightedSampleNeighborsRequest request;
    *request.mutable_node_ids() = {std::begin(node_ids), std::end(node_ids)};
    *request.mutable_edge_types() = {std::begin(edge_types), std::end(edge_types)};
    request.set_count(count);
    request.set_default_node_id(default_node_id);
    request.set_default_node_weight(default_weight);
    request.set_default_edge_type(default_edge_type);
    std::vector<std::future<void>> futures;
    std::vector<WeightedSampleNeighborsReply> replies(m_engine_stubs.size());

    // Cummulative total neighbor weights for each node.
    // We it to organize bernulli trials to merge node
    // neighbors that are split across shards.
    std::vector<float> shard_weights(node_ids.size());
    std::mutex mtx;
    for (size_t shard = 0; shard < m_engine_stubs.size(); ++shard)
    {
        request.set_seed(subseed(engine));

        auto *call = new AsyncClientCall();

        auto response_reader =
            m_engine_stubs[shard]->PrepareAsyncWeightedSampleNeighbors(&call->context, request, NextCompletionQueue());

        call->callback = [&reply = replies[shard], count, output_neighbors, output_types, output_weights, node_ids,
                          &mtx, &engine, &shard_weights, default_node_id, default_weight, default_edge_type]() {
            if (reply.node_ids().empty())
            {
                return;
            }

            auto curr_nodes = std::begin(node_ids);
            auto curr_out_neighbor = std::begin(output_neighbors);
            auto curr_out_type = std::begin(output_types);
            auto curr_out_weight = std::begin(output_weights);
            auto curr_shard_weight = std::begin(shard_weights);

            auto curr_reply_neighbor = std::begin(reply.neighbor_ids());
            auto curr_reply_type = std::begin(reply.neighbor_types());
            auto curr_reply_weight = std::begin(reply.neighbor_weights());
            auto curr_reply_shard_weight = std::begin(reply.shard_weights());
            boost::random::uniform_real_distribution<float> selector(0, 1);

            // We need to lock the merge in case some nodes are present in multiple
            // servers(a super node with lots of neighbors). To keep contention low
            // we'll use a global lock per response instead of per node.
            std::lock_guard guard(mtx);

            // The strategy is to zip nodes from server response matching to the input
            // nodes.
            for (const auto &reply_node_id : reply.node_ids())
            {
                // Loop until we find a match
                for (; curr_nodes != std::end(node_ids) && *curr_nodes != reply_node_id; ++curr_nodes)
                {
                    curr_out_neighbor += count;
                    curr_out_weight += count;
                    curr_out_type += count;
                    ++curr_shard_weight;
                }

                *curr_shard_weight += *curr_reply_shard_weight;
                if (*curr_shard_weight == 0)
                {
                    ++curr_shard_weight;
                    ++curr_reply_shard_weight;

                    curr_out_neighbor = std::fill_n(curr_out_neighbor, count, default_node_id);
                    curr_out_weight = std::fill_n(curr_out_weight, count, default_weight);
                    curr_out_type = std::fill_n(curr_out_type, count, default_edge_type);

                    curr_reply_neighbor += count;
                    curr_reply_type += count;
                    curr_reply_weight += count;
                    ++curr_nodes;
                    continue;
                }

                float overwrite_rate = *curr_reply_shard_weight / *curr_shard_weight;
                for (size_t i = 0; i < count; ++i)
                {
                    // Perf optimization for the most common scenario: every node has its
                    // neighbors in one partition.
                    if (overwrite_rate < 1.0f && selector(engine) > overwrite_rate)
                    {
                        ++curr_reply_weight;
                        ++curr_reply_neighbor;
                        ++curr_reply_type;
                        ++curr_out_neighbor;
                        ++curr_out_type;
                        ++curr_out_weight;
                        continue;
                    }

                    *(curr_out_neighbor++) = *(curr_reply_neighbor++);
                    *(curr_out_type++) = *(curr_reply_type++);
                    *(curr_out_weight++) = *(curr_reply_weight++);
                }

                ++curr_reply_shard_weight;
                ++curr_shard_weight;
                ++curr_nodes;
            }

            assert(curr_reply_weight == std::end(reply.neighbor_weights()));
            assert(curr_reply_neighbor == std::end(reply.neighbor_ids()));
            assert(curr_reply_type == std::end(reply.neighbor_types()));
            assert(curr_reply_shard_weight == std::end(reply.shard_weights()));
        };

        futures.emplace_back(call->promise.get_future());
        response_reader->StartCall();
        response_reader->Finish(&replies[shard], &call->status, static_cast<void *>(call));
    }

    WaitForFutures(futures);
}

void GRPCClient::UniformSampleNeighbor(bool without_replacement, int64_t seed, std::span<const NodeId> node_ids,
                                       std::span<const Type> edge_types, size_t count,
                                       std::span<NodeId> output_neighbors, std::span<Type> output_types,
                                       NodeId default_node_id, Type default_type)
{
    snark::Xoroshiro128PlusGenerator engine(seed);
    boost::random::uniform_int_distribution<int64_t> subseed(std::numeric_limits<int64_t>::min(),
                                                             std::numeric_limits<int64_t>::max());

    UniformSampleNeighborsRequest request;
    *request.mutable_node_ids() = {std::begin(node_ids), std::end(node_ids)};
    *request.mutable_edge_types() = {std::begin(edge_types), std::end(edge_types)};
    request.set_count(count);
    request.set_default_node_id(default_node_id);
    request.set_default_edge_type(default_type);
    request.set_without_replacement(without_replacement);
    std::vector<std::future<void>> futures;
    std::vector<UniformSampleNeighborsReply> replies(m_engine_stubs.size());

    // Cummulative total neighbor weights for each node.
    // We it to organize bernulli trials to merge node
    // neighbors that are split across shards.
    std::vector<size_t> shard_counts(node_ids.size());
    std::mutex mtx;
    for (size_t shard = 0; shard < m_engine_stubs.size(); ++shard)
    {
        request.set_seed(subseed(engine));

        auto *call = new AsyncClientCall();

        auto response_reader =
            m_engine_stubs[shard]->PrepareAsyncUniformSampleNeighbors(&call->context, request, NextCompletionQueue());
        call->callback = [&reply = replies[shard], count, output_types, node_ids, &mtx, &engine, &shard_counts,
                          output_neighbors, default_node_id, default_type]() {
            if (reply.node_ids().empty())
            {
                return;
            }

            auto curr_nodes = std::begin(node_ids);
            auto curr_out_neighbor = std::begin(output_neighbors);
            auto curr_out_type = std::begin(output_types);
            auto curr_shard_weight = std::begin(shard_counts);
            auto curr_reply_neighbor = std::begin(reply.neighbor_ids());
            auto curr_reply_type = std::begin(reply.neighbor_types());
            auto curr_reply_shard_weight = std::begin(reply.shard_counts());
            boost::random::uniform_real_distribution<float> selector(0, 1);

            // We need to lock the merge in case some nodes are present in multiple
            // servers(a super node with lots of neighbors). To keep contention low
            // we'll use a global lock per response instead of per node.
            std::lock_guard guard(mtx);

            // The strategy is to zip nodes from server response matching to the input
            // nodes.
            for (const auto &reply_node_id : reply.node_ids())
            {
                // Loop until we find a match
                for (; curr_nodes != std::end(node_ids) && *curr_nodes != reply_node_id; ++curr_nodes)
                {
                    curr_out_neighbor += count;
                    curr_out_type += count;
                    ++curr_shard_weight;
                }

                *curr_shard_weight += *curr_reply_shard_weight;
                if (*curr_shard_weight == 0)
                {
                    ++curr_shard_weight;
                    ++curr_reply_shard_weight;

                    curr_out_neighbor = std::fill_n(curr_out_neighbor, count, default_node_id);
                    curr_out_type = std::fill_n(curr_out_type, count, default_type);

                    curr_reply_neighbor += count;
                    curr_reply_type += count;
                    ++curr_nodes;
                    continue;
                }

                float overwrite_rate = float(*curr_reply_shard_weight) / *curr_shard_weight;
                for (size_t i = 0; i < count; ++i)
                {
                    // Perf optimization for the most common scenario: every node has its
                    // neighbors in one partition.
                    if (overwrite_rate < 1.0f && selector(engine) > overwrite_rate)
                    {
                        ++curr_reply_neighbor;
                        ++curr_reply_type;
                        ++curr_out_neighbor;
                        ++curr_out_type;
                        continue;
                    }

                    *(curr_out_neighbor++) = *(curr_reply_neighbor++);
                    *(curr_out_type++) = *(curr_reply_type++);
                }

                ++curr_reply_shard_weight;
                ++curr_shard_weight;
                ++curr_nodes;
            }

            assert(curr_reply_neighbor == std::end(reply.neighbor_ids()));
            assert(curr_reply_type == std::end(reply.neighbor_types()));
            assert(curr_reply_shard_weight == std::end(reply.shard_counts()));
        };

        futures.emplace_back(call->promise.get_future());
        response_reader->StartCall();
        response_reader->Finish(&replies[shard], &call->status, static_cast<void *>(call));
    }

    WaitForFutures(futures);
}

uint64_t GRPCClient::CreateSampler(bool is_edge, CreateSamplerRequest_Category category, std::span<Type> types)
{
    snark::CreateSamplerRequest request;
    *request.mutable_enitity_types() = {std::begin(types), std::end(types)};
    request.set_is_edge(is_edge);
    request.set_category(category);

    std::vector<std::future<void>> futures;
    std::vector<CreateSamplerReply> replies(m_sampler_stubs.size());

    std::vector<uint64_t> sub_sampler_ids(m_sampler_stubs.size());
    std::vector<float> sub_sampler_weights(m_sampler_stubs.size());
    for (size_t shard = 0; shard < m_sampler_stubs.size(); ++shard)
    {
        auto *call = new AsyncClientCall();

        auto response_reader =
            m_sampler_stubs[shard]->PrepareAsyncCreate(&call->context, request, NextCompletionQueue());

        call->callback = [&reply = replies[shard], &sub_sampler_id = sub_sampler_ids[shard],
                          &sub_sampler_weight = sub_sampler_weights[shard]]() {
            sub_sampler_id = reply.sampler_id();
            sub_sampler_weight = reply.weight();
        };

        futures.emplace_back(call->promise.get_future());
        response_reader->StartCall();
        response_reader->Finish(&replies[shard], &call->status, static_cast<void *>(call));
    }

    WaitForFutures(futures);

    // Adjust weights for future sampling.
    conditional_probabilities(sub_sampler_weights);
    std::lock_guard l(m_sampler_mutex);
    uint64_t sampler_id = m_sampler_ids.size();
    m_sampler_ids.emplace_back(std::move(sub_sampler_ids));
    m_sampler_weights.emplace_back(std::move(sub_sampler_weights));
    return sampler_id;
}

void GRPCClient::SampleNodes(int64_t seed, uint64_t sampler_id, std::span<NodeId> out_node_ids,
                             std::span<Type> out_types)
{
    snark::SampleRequest request;
    request.set_is_edge(false);

    std::vector<std::future<void>> futures;
    std::vector<SampleReply> replies(m_sampler_stubs.size());

    std::span<const float> weights;
    std::span<const uint64_t> sampler_ids;
    {
        std::lock_guard l(m_sampler_mutex);
        weights = std::span(m_sampler_weights[sampler_id].data(), m_sampler_stubs.size());

        sampler_ids = std::span(m_sampler_ids[sampler_id].data(), m_sampler_stubs.size());
    }

    snark::Xoroshiro128PlusGenerator gen(seed);
    boost::random::uniform_int_distribution<int64_t> subseed;
    size_t left = out_types.size();
    size_t position = 0;

    for (size_t shard = 0; shard < m_sampler_stubs.size(); ++shard)
    {
        if (sampler_ids[shard] == empty_sampler_id)
        {
            continue;
        }

        auto *call = new AsyncClientCall();
        const size_t element_count = boost::random::binomial_distribution<int32_t>(left, weights[shard])(gen);
        left -= element_count;

        request.set_seed(subseed(gen));
        request.set_sampler_id(sampler_ids[shard]);
        request.set_count(element_count);
        auto response_reader =
            m_sampler_stubs[shard]->PrepareAsyncSample(&call->context, request, NextCompletionQueue());
        call->callback = [&reply = replies[shard], types = out_types.subspan(position, element_count),
                          nodes = out_node_ids.subspan(position, element_count)]() {
            std::copy(std::begin(reply.types()), std::end(reply.types()), std::begin(types));

            std::copy(std::begin(reply.node_ids()), std::end(reply.node_ids()), std::begin(nodes));
        };

        position += element_count;
        futures.emplace_back(call->promise.get_future());
        response_reader->StartCall();
        response_reader->Finish(&replies[shard], &call->status, static_cast<void *>(call));
    }

    WaitForFutures(futures);
}

void GRPCClient::SampleEdges(int64_t seed, uint64_t sampler_id, std::span<NodeId> out_src_node_ids,
                             std::span<Type> out_types, std::span<NodeId> out_dst_node_ids)
{
    snark::SampleRequest request;
    request.set_is_edge(true);

    std::vector<std::future<void>> futures;
    std::vector<SampleReply> replies(m_sampler_stubs.size());

    std::span<const float> weights;
    std::span<const uint64_t> sampler_ids;
    {
        std::lock_guard l(m_sampler_mutex);
        weights = std::span(m_sampler_weights[sampler_id].data(), m_sampler_stubs.size());

        sampler_ids = std::span(m_sampler_ids[sampler_id].data(), m_sampler_stubs.size());
    }

    snark::Xoroshiro128PlusGenerator gen(seed);
    boost::random::uniform_int_distribution<int64_t> subseed;
    size_t left = out_types.size();
    size_t position = 0;

    for (size_t shard = 0; shard < m_sampler_stubs.size() && left > 0; ++shard)
    {
        auto *call = new AsyncClientCall();
        const size_t shard_count = boost::random::binomial_distribution<int32_t>(left, weights[shard])(gen);
        left -= shard_count;

        request.set_seed(subseed(gen));
        request.set_sampler_id(sampler_ids[shard]);
        request.set_count(shard_count);

        auto response_reader =
            m_sampler_stubs[shard]->PrepareAsyncSample(&call->context, request, NextCompletionQueue());
        call->callback = [&reply = replies[shard], shard_count, types = out_types.subspan(position, shard_count),
                          src_nodes = out_src_node_ids.subspan(position, shard_count),
                          dst_nodes = out_dst_node_ids.subspan(position, shard_count)]() {
            std::copy(std::begin(reply.types()), std::end(reply.types()), std::begin(types));
            std::copy_n(std::begin(reply.node_ids()), shard_count, std::begin(src_nodes));
            std::copy_n(std::begin(reply.node_ids()) + shard_count, shard_count, std::begin(dst_nodes));
        };

        position += shard_count;
        futures.emplace_back(call->promise.get_future());
        response_reader->StartCall();
        response_reader->Finish(&replies[shard], &call->status, static_cast<void *>(call));
    }

    WaitForFutures(futures);
}

void GRPCClient::WriteMetadata(std::filesystem::path path)
{
    EmptyMessage request;
    MetadataReply reply;
    auto *call = new AsyncClientCall();
    auto response_reader =
        m_engine_stubs.front()->PrepareAsyncGetMetadata(&call->context, request, NextCompletionQueue());

    call->callback = [&reply, &path]() {
        Metadata meta;
        meta.m_version = reply.version();
        meta.m_node_count = reply.nodes();
        meta.m_edge_count = reply.edges();
        meta.m_node_type_count = reply.node_types();
        meta.m_edge_type_count = reply.edge_types();
        meta.m_partition_count = reply.partitions();
        meta.m_node_feature_count = reply.node_features();
        meta.m_edge_feature_count = reply.edge_features();
        meta.m_node_count_per_type = {std::begin(reply.node_count_per_type()), std::end(reply.node_count_per_type())};
        meta.m_edge_count_per_type = {std::begin(reply.edge_count_per_type()), std::end(reply.edge_count_per_type())};

        auto curr_node_type = std::begin(reply.node_partition_weights());
        auto curr_edge_type = std::begin(reply.edge_partition_weights());
        meta.m_partition_node_weights.reserve(meta.m_partition_count);
        meta.m_partition_edge_weights.reserve(meta.m_partition_count);
        for (size_t partition = 0; partition < meta.m_partition_count; ++partition)
        {
            meta.m_partition_node_weights.emplace_back();
            auto &node_weights = meta.m_partition_node_weights.back();
            node_weights.reserve(meta.m_node_type_count);
            std::copy_n(curr_node_type, meta.m_node_type_count, std::back_inserter(node_weights));
            curr_node_type += meta.m_node_type_count;

            meta.m_partition_edge_weights.emplace_back();
            auto &edge_weights = meta.m_partition_edge_weights.back();
            edge_weights.reserve(meta.m_edge_type_count);
            std::copy_n(curr_edge_type, meta.m_edge_type_count, std::back_inserter(edge_weights));
            curr_edge_type += meta.m_edge_type_count;
        }

        meta.Write(path.string().c_str());
    };

    // Order is important: call variable can be deleted before futures can be
    // obtained.
    auto future = call->promise.get_future();
    response_reader->StartCall();
    response_reader->Finish(&reply, &call->status, static_cast<void *>(call));
    future.get();
}

grpc::CompletionQueue *GRPCClient::NextCompletionQueue()
{
    return &m_completion_queue[m_counter++ % m_completion_queue.size()];
}

GRPCClient::~GRPCClient()
{
    for (auto &q : m_completion_queue)
    {
        q.Shutdown();
    }
    for (auto &t : m_reply_threads)
    {
        t.join();
    }
}

} // namespace snark
