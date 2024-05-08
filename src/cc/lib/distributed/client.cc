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
#include <queue>
#include <random>
#include <thread>
#include <type_traits>

#include "src/cc/lib/distributed/call_data.h"
#include "src/cc/lib/graph/merger.h"
#include "src/cc/lib/graph/xoroshiro.h"

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

void WaitForFutures(std::vector<std::future<void>> &futures)
{
    for (auto &f : futures)
    {
        f.get();
    }
}

template <typename SparseRequest, typename NextCompletionQueue>
void GetSparseFeature(const SparseRequest &request,
                      std::vector<std::unique_ptr<snark::GraphEngine::Stub>> &engine_stubs, size_t input_size,
                      size_t feature_count, std::span<int64_t> out_dimensions,
                      std::vector<std::vector<int64_t>> &out_indices, std::vector<std::vector<uint8_t>> &out_values,
                      NextCompletionQueue next_completion_queue, std::shared_ptr<snark::Logger> logger)
{
    std::vector<std::future<void>> futures;
    futures.reserve(engine_stubs.size());
    std::vector<snark::SparseFeaturesReply> replies(engine_stubs.size());
    std::vector<snark::SparseFeatureIndex> response_index(input_size * feature_count,
                                                          snark::SparseFeatureIndex{.shard = -1, .timestamp = -1});

    // We use mutex to protect response_index from concurrent access.
    // Locking happens per request, so it is not a bottleneck.
    std::mutex mtx;
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

        call->callback = [&reply = replies[shard], &response_index, shard, out_dimensions, feature_count, &mtx,
                          logger]() {
            snark::UpdateSparseFeatureIndex(reply, response_index, shard, out_dimensions, feature_count, mtx, logger);
        };
        // Order is important: call variable can be deleted before futures can be
        // obtained.
        futures.emplace_back(call->promise.get_future());
        response_reader->StartCall();
        response_reader->Finish(&replies[shard], &call->status, static_cast<void *>(call));
    }

    WaitForFutures(futures);
    snark::ExtractFeatures(response_index, replies, out_dimensions, out_indices, out_values, input_size);
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
    std::vector<std::tuple<size_t, size_t, snark::Timestamp>> response_index(input_size * feature_count, {0, 0, -1});
    std::mutex mtx;

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

        call->callback = [&reply = replies[shard], &response_index, shard, out_dimensions, &mtx]() {
            UpdateStringFeatureIndex(reply, mtx, response_index, shard, out_dimensions);
        };

        // Order is important: call variable can be deleted before futures can be
        // obtained.
        futures.emplace_back(call->promise.get_future());
        response_reader->StartCall();
        response_reader->Finish(&replies[shard], &call->status, static_cast<void *>(call));
    }

    WaitForFutures(futures);
    snark::ExtractStringFeatures(response_index, replies, out_dimensions, out_values);
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
                       uint32_t num_threads_per_cq, std::shared_ptr<Logger> logger)
{
    if (!logger)
    {
        logger = std::make_shared<GLogger>();
    }
    m_logger = logger;
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
    return [&queue = m_completion_queue[index], logger = m_logger]() {
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
                    logger->log_error("Client failed to process request. Exception: %s", e.what());
                    call->promise.set_exception(std::current_exception());
                }
            }
            else
            {
                logger->log_error("Request failed, code: %d. Message: %s", call->status.error_code(),
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

void GRPCClient::GetNodeFeature(std::span<const NodeId> node_ids, std::span<const snark::Timestamp> timestamps,
                                std::span<FeatureMeta> features, std::span<uint8_t> output)
{
    assert(std::accumulate(std::begin(features), std::end(features), size_t(0),
                           [](size_t val, const auto &f) { return val + f.second; }) *
               node_ids.size() ==
           output.size());

    NodeFeaturesRequest request;
    const auto node_len = node_ids.size();
    *request.mutable_node_ids() = {std::begin(node_ids), std::end(node_ids)};
    *request.mutable_timestamps() = {std::begin(timestamps), std::end(timestamps)};
    size_t curr_feature_prefix = 0;
    std::vector<size_t> feature_len_prefix;
    feature_len_prefix.reserve(features.size());

    for (const auto &feature : features)
    {
        auto wire_feature = request.add_features();
        wire_feature->set_id(feature.first);
        wire_feature->set_size(feature.second);
        feature_len_prefix.emplace_back(curr_feature_prefix);
        curr_feature_prefix += feature.second;
    }
    const size_t fv_size = output.size() / node_len;
    std::vector<std::future<void>> futures;
    futures.reserve(m_engine_stubs.size());
    std::vector<NodeFeaturesReply> replies(m_engine_stubs.size());
    std::mutex mtx;
    std::vector<std::tuple<int32_t, int32_t, snark::Timestamp>> response_index(node_ids.size() * features.size(),
                                                                               {-1, -1, -1});
    for (size_t shard = 0; shard < m_engine_stubs.size(); ++shard)
    {
        auto *call = new AsyncClientCall();

        auto response_reader =
            m_engine_stubs[shard]->PrepareAsyncGetNodeFeatures(&call->context, request, NextCompletionQueue());

        call->callback = [&reply = replies[shard], shard, &mtx, &features, &response_index]() {
            if (reply.offsets().empty())
            {
                return;
            }

            std::lock_guard lock(mtx);
            int32_t timestamp_index = 0;
            for (int32_t node_index = 0; node_index < reply.offsets().size(); ++node_index)
            {
                const auto index = reply.offsets(node_index) * features.size();
                for (int32_t feature_index = 0; feature_index < int32_t(features.size()); ++feature_index)
                {
                    const auto ts = reply.timestamps(timestamp_index);

                    // pick latest value across all shards.
                    if (std::get<2>(response_index[index + feature_index]) <= ts)
                    {
                        response_index[index + feature_index] =
                            std::make_tuple(int32_t(shard), int32_t(node_index), ts);
                    }

                    ++timestamp_index;
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

    for (size_t i = 0; i < response_index.size(); ++i)
    {
        snark::Timestamp ts;
        int32_t node_index, shard_index;
        std::tie(shard_index, node_index, ts) = response_index[i];
    }

    auto values = std::begin(output);
    for (size_t node_index = 0; node_index < node_len; ++node_index)
    {
        for (size_t feature_index = 0; feature_index < features.size(); ++feature_index)
        {
            snark::Timestamp ts;
            int32_t node_sub_index, shard_index;

            std::tie(shard_index, node_sub_index, ts) = response_index[node_index * features.size() + feature_index];
            if (ts >= 0)
            {
                // Use c_str since string iterators can process wide charachters on windows.
                auto reply_feature_values = replies[shard_index].feature_values().c_str();
                values =
                    std::copy_n(reply_feature_values + node_sub_index * fv_size + feature_len_prefix[feature_index],
                                features[feature_index].second, values);
            }
            else
            {
                values = std::fill_n(values, features[feature_index].second, 0);
            }
        }
    }
}

void GRPCClient::GetEdgeFeature(std::span<const NodeId> edge_src_ids, std::span<const NodeId> edge_dst_ids,
                                std::span<const Type> edge_types, std::span<const snark::Timestamp> timestamps,
                                std::span<FeatureMeta> features, std::span<uint8_t> output)
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
    *request.mutable_timestamps() = {std::begin(timestamps), std::end(timestamps)};

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

void GRPCClient::GetNodeSparseFeature(std::span<const NodeId> node_ids, std::span<const snark::Timestamp> timestamps,
                                      std::span<const FeatureId> features, std::span<int64_t> out_dimensions,
                                      std::vector<std::vector<int64_t>> &out_indices,
                                      std::vector<std::vector<uint8_t>> &out_values)
{
    assert(out_indices.size() == features.size());
    assert(out_dimensions.size() == features.size());

    // Fill out_dimensions in case nodes don't have some features.
    std::fill(std::begin(out_dimensions), std::end(out_dimensions), 0);
    NodeSparseFeaturesRequest request;
    *request.mutable_node_ids() = {std::begin(node_ids), std::end(node_ids)};
    *request.mutable_feature_ids() = {std::begin(features), std::end(features)};
    *request.mutable_timestamps() = {std::begin(timestamps), std::end(timestamps)};

    GetSparseFeature(request, m_engine_stubs, node_ids.size(), features.size(), out_dimensions, out_indices, out_values,
                     std::bind(&GRPCClient::NextCompletionQueue, this), m_logger);
}

void GRPCClient::GetEdgeSparseFeature(std::span<const NodeId> edge_src_ids, std::span<const NodeId> edge_dst_ids,
                                      std::span<const Type> edge_types, std::span<const snark::Timestamp> timestamps,
                                      std::span<const FeatureId> features, std::span<int64_t> out_dimensions,
                                      std::vector<std::vector<int64_t>> &out_indices,
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
    *request.mutable_timestamps() = {std::begin(timestamps), std::end(timestamps)};

    GetSparseFeature(request, m_engine_stubs, len, features.size(), out_dimensions, out_indices, out_values,
                     std::bind(&GRPCClient::NextCompletionQueue, this), m_logger);
}

void GRPCClient::GetNodeStringFeature(std::span<const NodeId> node_ids, std::span<const snark::Timestamp> timestamps,
                                      std::span<const FeatureId> features, std::span<int64_t> out_dimensions,
                                      std::vector<uint8_t> &out_values)
{
    NodeSparseFeaturesRequest request;
    *request.mutable_node_ids() = {std::begin(node_ids), std::end(node_ids)};
    *request.mutable_feature_ids() = {std::begin(features), std::end(features)};
    *request.mutable_timestamps() = {std::begin(timestamps), std::end(timestamps)};
    GetStringFeature(request, m_engine_stubs, node_ids.size(), features.size(), out_dimensions, out_values,
                     std::bind(&GRPCClient::NextCompletionQueue, this));
}

void GRPCClient::GetEdgeStringFeature(std::span<const NodeId> edge_src_ids, std::span<const NodeId> edge_dst_ids,
                                      std::span<const Type> edge_types, std::span<const snark::Timestamp> timestamps,
                                      std::span<const FeatureId> features, std::span<int64_t> out_dimensions,
                                      std::vector<uint8_t> &out_values)
{
    const auto len = edge_types.size();
    assert(len == edge_src_ids.size());
    assert(len == edge_dst_ids.size());

    EdgeSparseFeaturesRequest request;
    *request.mutable_node_ids() = {std::begin(edge_src_ids), std::end(edge_src_ids)};
    request.mutable_node_ids()->Add(std::begin(edge_dst_ids), std::end(edge_dst_ids));
    request.mutable_types()->Add(std::begin(edge_types), std::end(edge_types));
    *request.mutable_feature_ids() = {std::begin(features), std::end(features)};
    *request.mutable_timestamps() = {std::begin(timestamps), std::end(timestamps)};

    GetStringFeature(request, m_engine_stubs, len, features.size(), out_dimensions, out_values,
                     std::bind(&GRPCClient::NextCompletionQueue, this));
}

void GRPCClient::NeighborCount(std::span<const NodeId> node_ids, std::span<const Type> edge_types,
                               std::span<const snark::Timestamp> timestamps, std::span<uint64_t> output_neighbor_counts)
{
    GetNeighborsRequest request;

    *request.mutable_node_ids() = {std::begin(node_ids), std::end(node_ids)};
    *request.mutable_edge_types() = {std::begin(edge_types), std::end(edge_types)};
    *request.mutable_timestamps() = {std::begin(timestamps), std::end(timestamps)};

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

void GRPCClient::LastNCreated(bool return_edge_created_ts, std::span<const NodeId> input_node_ids,
                              std::span<Type> input_edge_types, std::span<const Timestamp> input_timestamps,
                              size_t count, std::span<NodeId> output_neighbor_ids,
                              std::span<Type> output_neighbor_types, std::span<float> neighbors_weights,
                              std::span<Timestamp> output_timestamps, NodeId default_node_id, float default_weight,
                              Type default_edge_type, Timestamp default_timestamp)
{
    GetLastNCreatedNeighborsRequest request;

    *request.mutable_node_ids() = {std::begin(input_node_ids), std::end(input_node_ids)};
    *request.mutable_edge_types() = {std::begin(input_edge_types), std::end(input_edge_types)};
    *request.mutable_timestamps() = {std::begin(input_timestamps), std::end(input_timestamps)};
    request.set_count(count);

    std::vector<std::future<void>> futures;
    std::vector<GetNeighborsReply> replies(std::size(m_engine_stubs));
    std::atomic<size_t> responses_left{std::size(m_engine_stubs)};

    for (size_t shard = 0; shard < m_engine_stubs.size(); ++shard)
    {
        auto *call = new AsyncClientCall();
        auto response_reader =
            m_engine_stubs[shard]->PrepareAsyncGetLastNCreatedNeighbors(&call->context, request, NextCompletionQueue());

        call->callback = [&responses_left, &replies, input_node_ids, output_neighbor_ids, output_neighbor_types,
                          neighbors_weights, output_timestamps, count, default_node_id, default_weight,
                          default_edge_type, default_timestamp, return_edge_created_ts]() {
            MergeLastNCreatedNeighbors(count, replies, input_node_ids, responses_left, output_neighbor_ids,
                                       output_neighbor_types, neighbors_weights, output_timestamps, default_node_id,
                                       default_weight, default_edge_type, default_timestamp, return_edge_created_ts);
        };

        futures.emplace_back(call->promise.get_future());
        response_reader->StartCall();
        response_reader->Finish(&replies[shard], &call->status, static_cast<void *>(call));
    }

    WaitForFutures(futures);
}

void GRPCClient::FullNeighbor(bool return_edge_created_ts, std::span<const NodeId> node_ids,
                              std::span<const Type> edge_types, std::span<const snark::Timestamp> timestamps,
                              std::vector<NodeId> &output_nodes, std::vector<Type> &output_types,
                              std::vector<float> &output_weights, std::vector<Timestamp> &out_edge_created_ts,
                              std::span<uint64_t> output_neighbor_counts)
{
    GetNeighborsRequest request;

    *request.mutable_node_ids() = {std::begin(node_ids), std::end(node_ids)};
    *request.mutable_edge_types() = {std::begin(edge_types), std::end(edge_types)};
    request.set_return_edge_created_ts(return_edge_created_ts);
    *request.mutable_timestamps() = {std::begin(timestamps), std::end(timestamps)};
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
                          &output_neighbor_counts, &reply_offsets, &out_edge_created_ts, return_edge_created_ts,
                          logger = m_logger]() {
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
                        logger->log_error(
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
                    if (return_edge_created_ts)
                    {
                        auto edge_creation_ts_start = reply.timestamps().begin() + offset;
                        out_edge_created_ts.insert(std::end(out_edge_created_ts), edge_creation_ts_start,
                                                   edge_creation_ts_start + count);
                    }

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

void GRPCClient::WeightedSampleNeighbor(bool return_edge_created_ts, int64_t seed, std::span<const NodeId> node_ids,
                                        std::span<const Type> edge_types, std::span<const snark::Timestamp> timestamps,
                                        size_t count, std::span<NodeId> output_neighbors, std::span<Type> output_types,
                                        std::span<float> output_weights,
                                        std::span<snark::Timestamp> output_edge_created_ts, NodeId default_node_id,
                                        float default_weight, Type default_edge_type)
{
    snark::Xoroshiro128PlusGenerator engine(seed);
    boost::random::uniform_int_distribution<int64_t> subseed(std::numeric_limits<int64_t>::min(),
                                                             std::numeric_limits<int64_t>::max());

    WeightedSampleNeighborsRequest request;
    *request.mutable_node_ids() = {std::begin(node_ids), std::end(node_ids)};
    *request.mutable_edge_types() = {std::begin(edge_types), std::end(edge_types)};
    *request.mutable_timestamps() = {std::begin(timestamps), std::end(timestamps)};
    request.set_return_edge_created_ts(return_edge_created_ts);
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

        call->callback = [&reply = replies[shard], count, output_neighbors, output_types, output_weights,
                          output_edge_created_ts, node_ids, &mtx, &engine, &shard_weights, default_node_id,
                          default_weight, default_edge_type, return_edge_created_ts]() {
            MergeWeightedSampledNeighbors(reply, mtx, count, output_neighbors, output_types, output_weights,
                                          output_edge_created_ts, node_ids, engine, shard_weights, default_node_id,
                                          default_weight, default_edge_type, return_edge_created_ts);
        };

        futures.emplace_back(call->promise.get_future());
        response_reader->StartCall();
        response_reader->Finish(&replies[shard], &call->status, static_cast<void *>(call));
    }

    WaitForFutures(futures);
}

void GRPCClient::UniformSampleNeighbor(bool without_replacement, bool return_edge_created_ts, int64_t seed,
                                       std::span<const NodeId> node_ids, std::span<const Type> edge_types,
                                       std::span<const snark::Timestamp> timestamps, size_t count,
                                       std::span<NodeId> output_neighbors, std::span<Type> output_types,
                                       std::span<snark::Timestamp> output_edge_created_ts, NodeId default_node_id,
                                       Type default_type)
{
    snark::Xoroshiro128PlusGenerator engine(seed);
    boost::random::uniform_int_distribution<int64_t> subseed(std::numeric_limits<int64_t>::min(),
                                                             std::numeric_limits<int64_t>::max());

    UniformSampleNeighborsRequest request;
    *request.mutable_node_ids() = {std::begin(node_ids), std::end(node_ids)};
    *request.mutable_edge_types() = {std::begin(edge_types), std::end(edge_types)};
    *request.mutable_timestamps() = {std::begin(timestamps), std::end(timestamps)};
    request.set_return_edge_created_ts(return_edge_created_ts);
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
        call->callback = []() {};
        futures.emplace_back(call->promise.get_future());
        response_reader->StartCall();
        response_reader->Finish(&replies[shard], &call->status, static_cast<void *>(call));
    }

    WaitForFutures(futures);
    std::fill(std::begin(output_neighbors), std::end(output_neighbors), default_node_id);
    std::fill(std::begin(output_types), std::end(output_types), default_type);
    if (return_edge_created_ts)
    {
        std::fill(std::begin(output_edge_created_ts), std::end(output_edge_created_ts), snark::PLACEHOLDER_TIMESTAMP);
    }

    // Node ids in responses have the same order as in the request. We can zip them together in linear time,
    // rather than iterating through all responses for each node.
    using ResponseIterator = google::protobuf::RepeatedField<int64_t>::const_iterator;
    std::vector<ResponseIterator> node_id_iterators(replies.size());
    for (size_t i = 0; i < replies.size(); ++i)
    {
        node_id_iterators[i] = replies[i].node_ids().begin();
    }

    WithoutReplacementMerge merger_without_replacement(count, engine);
    WithReplacement merger_with_replacement(count, engine);
    for (size_t node_index = 0; node_index < node_ids.size(); ++node_index)
    {
        if (without_replacement)
        {
            merger_without_replacement.reset();
        }
        else
        {
            merger_with_replacement.reset();
        }
        const auto node_id = node_ids[node_index];
        auto neighbor_reservoir = output_neighbors.subspan(node_index * count, count);
        auto type_reservoir = output_types.subspan(node_index * count, count);
        std::span<snark::Timestamp> edge_created_ts_reservoir;
        if (return_edge_created_ts)
        {
            edge_created_ts_reservoir = output_edge_created_ts.subspan(node_index * count, count);
        }
        for (size_t reply_index = 0; reply_index < replies.size(); ++reply_index)
        {
            const auto &reply = replies[reply_index];
            if (node_id_iterators[reply_index] == reply.node_ids().end() || *node_id_iterators[reply_index] != node_id)
            {
                continue;
            }

            const auto &node_id_iterator = node_id_iterators[reply_index];
            const auto reply_node_offset = node_id_iterator - reply.node_ids().begin();
            const auto neighbors_offset = count * reply_node_offset;
            const auto reply_weight = reply.shard_counts(reply_node_offset);
            auto update = [&reply, neighbor_reservoir, type_reservoir, edge_created_ts_reservoir, count,
                           neighbors_offset, return_edge_created_ts](size_t pick, size_t offset) {
                // In case of merging neighbors from larger universe, we'll have to normalize by count.
                auto reply_offset = neighbors_offset + (offset % count);
                neighbor_reservoir[pick] = reply.neighbor_ids(reply_offset);
                type_reservoir[pick] = reply.neighbor_types(reply_offset);
                if (return_edge_created_ts)
                {
                    edge_created_ts_reservoir[pick] = reply.timestamps(reply_offset);
                }
            };
            if (without_replacement)
            {
                merger_without_replacement.add(reply_weight, update);
            }
            else
            {
                merger_with_replacement.add(reply_weight, update);
            }
            ++node_id_iterators[reply_index];
        }
    }
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
        meta.m_watermark = reply.watermark();
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
