// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

/* This header can be read by both C and C++ compilers */
#ifndef PYGRAPH_H
#define PYGRAPH_H

#ifdef _WIN32
#define DEEPGNN_DLL __declspec(dllexport)
#else
#define DEEPGNN_DLL __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
#include <cstddef>
#include <cstdint>
#include <memory>

namespace snark
{
class Sampler;
class GRPCServer;
} // namespace snark

namespace deep_graph
{
namespace python
{

struct GraphInternal;

struct PyGraph
{
    std::unique_ptr<GraphInternal> graph;
};
struct PySampler
{
    std::unique_ptr<snark::Sampler> sampler;
};
struct PyServer
{
    std::unique_ptr<snark::GRPCServer> server;
};

enum PyPartitionStorageType // C interface to PartitionStoragetype in types.h
{
    memory,
    disk,
};
#else

#include <stddef.h>
#include <stdint.h>
typedef struct PyGraph PyGraph;

typedef struct PySampler PySampler;

typedef struct PyServer PyServer;

enum PyPartitionStorageType // C interface to PartitionStoragetype in types.h
{
    memory,
    disk,
};
#endif

#ifdef __cplusplus
extern "C"
{
#endif
    typedef int64_t NodeID;
    typedef int64_t Timestamp;
    typedef int32_t Type;
    typedef int32_t Feature;

    typedef void (*GetNeighborsCallback)(const NodeID *, const float *, const Type *, const Timestamp *, size_t);
    typedef void (*GetSparseFeaturesCallback)(const int64_t **, size_t *, const uint8_t **, size_t *, int64_t *);
    typedef void (*GetStringFeaturesCallback)(size_t, const uint8_t *);

    DEEPGNN_DLL extern int32_t CreateLocalGraph(PyGraph *graph, const char *meta_location, size_t count,
                                                uint32_t *partition_indices, const char **partition_locations,
                                                PyPartitionStorageType storage_type, const char *config_path);

    DEEPGNN_DLL extern int32_t StartServer(PyServer *graph, const char *meta_location, size_t count,
                                           uint32_t *partition_indices, const char **partition_locations,
                                           const char *host_name, const char *ssl_key, const char *ssl_cert,
                                           const char *ssl_root, const PyPartitionStorageType storage_type,
                                           const char *config_path, bool skip_feature_loading,
                                           bool skip_watermark_loading);

    DEEPGNN_DLL extern int32_t CreateRemoteClient(PyGraph *graph, const char *output_folder, const char **connection,
                                                  size_t connection_count, const char *ssl_cert, size_t num_threads,
                                                  size_t num_threads_per_cq, size_t num_custom_args,
                                                  const char **custom_args_keys, const char **custom_args_values);

    DEEPGNN_DLL extern int32_t GetNodeType(PyGraph *graph, NodeID *node_ids, size_t node_ids_size, Type *output,
                                           Type default_type);
    DEEPGNN_DLL extern int32_t GetNodeFeature(PyGraph *graph, NodeID *node_ids, size_t node_ids_size,
                                              Timestamp *time_stamps, Feature *features, size_t features_size,
                                              uint8_t *output, size_t output_size);
    DEEPGNN_DLL extern int32_t GetNodeSparseFeature(PyGraph *graph, NodeID *node_ids, size_t node_ids_size,
                                                    Timestamp *time_stamps, Feature *features, size_t features_size,
                                                    GetSparseFeaturesCallback callback);
    DEEPGNN_DLL extern int32_t GetNodeStringFeature(PyGraph *graph, NodeID *node_ids, size_t node_ids_size,
                                                    Timestamp *time_stamps, Feature *features, size_t features_size,
                                                    int64_t *dimensions, GetStringFeaturesCallback callback);
    DEEPGNN_DLL extern int32_t GetEdgeFeature(PyGraph *graph, NodeID *edge_src_ids, NodeID *edge_dst_ids,
                                              Type *edge_types, size_t edge_size, Timestamp *time_stamps,
                                              Feature *features, size_t features_size, uint8_t *output,
                                              size_t output_size);
    DEEPGNN_DLL extern int32_t GetEdgeSparseFeature(PyGraph *graph, NodeID *edge_src_ids, NodeID *edge_dst_ids,
                                                    Type *edge_types, size_t edge_size, Timestamp *time_stamps,
                                                    Feature *features, size_t features_size,
                                                    GetSparseFeaturesCallback callback);
    DEEPGNN_DLL extern int32_t GetEdgeStringFeature(PyGraph *graph, NodeID *edge_src_ids, NodeID *edge_dst_ids,
                                                    Type *edge_types, size_t edge_size, Timestamp *time_stamps,
                                                    Feature *features, size_t features_size, int64_t *dimensions,
                                                    GetStringFeaturesCallback callback);
    DEEPGNN_DLL extern int32_t NeighborCount(PyGraph *py_graph, NodeID *in_node_ids, size_t in_node_ids_size,
                                             Timestamp *time_stamps, Type *in_edge_types, size_t in_edge_types_size,
                                             uint64_t *out_neighbor_counts);
    DEEPGNN_DLL extern int32_t GetNeighbors(PyGraph *graph, bool return_edge_created_ts, NodeID *in_node_ids,
                                            size_t in_node_ids_size, Timestamp *time_stamps, Type *in_edge_types,
                                            size_t in_edge_types_size, uint64_t *out_neighbor_counts,
                                            GetNeighborsCallback callback);
    DEEPGNN_DLL extern int32_t WeightedSampleNeighbor(PyGraph *graph, bool return_edge_created_ts, int64_t seed,
                                                      NodeID *in_node_ids, size_t in_node_ids_size, Type *in_edge_types,
                                                      size_t in_edge_types_size, Timestamp *time_stamps, size_t count,
                                                      NodeID *out_neighbor_ids, Type *out_types, float *out_weights,
                                                      Timestamp *out_created_ts, NodeID default_node_id,
                                                      float default_weight, Type default_edge_type);
    DEEPGNN_DLL extern int32_t UniformSampleNeighbor(PyGraph *graph, bool without_replacement,
                                                     bool return_edge_created_ts, int64_t seed, NodeID *in_node_ids,
                                                     size_t int_node_ids_size, Type *in_edge_types,
                                                     size_t in_edge_types_size, Timestamp *timestamps, size_t count,
                                                     NodeID *out_neighbor_ids, Type *out_types,
                                                     Timestamp *out_created_ts, NodeID default_node_id,
                                                     Type default_edge_type);

    DEEPGNN_DLL extern int32_t RandomWalk(PyGraph *graph, int64_t seed, float p, float q, NodeID default_node_id,
                                          NodeID *in_node_ids, size_t in_node_ids_size, Timestamp *timestamps,
                                          Type *in_edge_types, size_t in_edge_types_size, size_t walk_length,
                                          NodeID *out_node_ids);

    DEEPGNN_DLL extern int32_t PPRSampleNeighbor(PyGraph *graph, NodeID *in_node_ids, size_t in_node_ids_size,
                                                 Timestamp *timestamps, Type *in_edge_types, size_t in_edge_types_size,
                                                 size_t count, float alpha, float eps, NodeID default_node_id,
                                                 float default_weight, NodeID *out_neighbor_ids, float *out_weights);

    DEEPGNN_DLL extern int32_t LastNCreatedNeighbor(PyGraph *py_graph, bool return_edge_created_ts, NodeID *in_node_ids,
                                                    size_t in_node_ids_size, Type *in_edge_types,
                                                    size_t in_edge_types_size, Timestamp *timestamps, size_t count,
                                                    NodeID *out_neighbor_ids, Type *out_types, float *out_weights,
                                                    Timestamp *out_timestamps, NodeID default_node_id,
                                                    float default_weight, Type default_edge_type,
                                                    Timestamp default_timestamp);

    // TODO(alsamylk): sorted neighbors

    DEEPGNN_DLL extern int32_t CreateWeightedNodeSampler(PyGraph *graph, PySampler *node_sampler, size_t count,
                                                         int32_t *types);
    DEEPGNN_DLL extern int32_t CreateUniformNodeSampler(PyGraph *graph, PySampler *node_sampler, size_t count,
                                                        int32_t *types);
    DEEPGNN_DLL extern int32_t CreateUniformNodeSamplerWithoutReplacement(PyGraph *py, PySampler *node_sampler,
                                                                          size_t count, int32_t *types);

    DEEPGNN_DLL extern int32_t SampleNodes(PySampler *sampler, int64_t seed, size_t count, NodeID *out_nodes,
                                           Type *out_types);

    DEEPGNN_DLL extern int32_t CreateWeightedEdgeSampler(PyGraph *graph, PySampler *edge_sampler, size_t count,
                                                         int32_t *types);
    DEEPGNN_DLL extern int32_t CreateUniformEdgeSampler(PyGraph *graph, PySampler *edge_sampler, size_t count,
                                                        int32_t *types);
    DEEPGNN_DLL extern int32_t CreateUniformEdgeSamplerWithoutReplacement(PyGraph *graph, PySampler *edge_sampler,
                                                                          size_t count, int32_t *types);

    DEEPGNN_DLL extern int32_t SampleEdges(PySampler *sampler, int64_t seed, size_t count, NodeID *out_src_id,
                                           NodeID *out_dst_id, Type *out_type);

    DEEPGNN_DLL extern int32_t ResetSampler(PySampler *sampler);
    DEEPGNN_DLL extern int32_t ResetGraph(PyGraph *graph);
    DEEPGNN_DLL extern int32_t ResetServer(PyServer *graph);

    DEEPGNN_DLL extern int32_t HDFSMoveMeta(const char *filename_src, const char *filename_dst,
                                            const char *config_path);

#ifdef __cplusplus
}

} // python
} // deep_graph
#endif

#endif /*PYGRAPH*/
