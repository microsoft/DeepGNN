// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "src/cc/lib/graph/graph.h"
#include <filesystem>
#include <string>
#include <tuple>
#include <vector>

namespace TestGraph
{

using NeighborRecord = std::tuple<snark::NodeId, snark::Type, float>;
struct Node
{
    int64_t m_id;
    int32_t m_type;
    float m_weight;
    std::vector<std::vector<float>> m_float_features; // first dimension is id, second is feature vector.
    std::vector<NeighborRecord> m_neighbors;

    // ordered in the same way as m_neighbors, 1st dimension is edge, 2nd - feature_id, 3rd actual data.
    std::vector<std::vector<std::vector<float>>> m_edge_features;
};

struct MemoryGraph
{
    std::vector<Node> m_nodes;

    // Temporal information.
    snark::Timestamp m_watermark = -1; // use -1 to flag a non-temporal graph.
    std::vector<std::pair<snark::Timestamp, snark::Timestamp>> m_edge_timestamps;
};

snark::Partition convert(std::filesystem::path path, std::string suffix, MemoryGraph t, size_t node_types);

std::vector<float> serialize_temporal_features(std::vector<snark::Timestamp> timestamps,
                                               std::vector<std::vector<float>> features);
} // namespace TestGraph
