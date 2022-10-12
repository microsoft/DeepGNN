// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#ifndef SNARK_METADATA_H
#define SNARK_METADATA_H

#include <filesystem>
#include <string>
#include <vector>

namespace snark
{

const size_t MINIMUM_SUPPORTED_VERSION = 1;

struct Metadata
{
    Metadata() = default;
    explicit Metadata(std::filesystem::path path, std::string config_path = "");
    void Write(std::filesystem::path path) const;

    // Graph information.
    size_t m_version;
    size_t m_node_count;
    size_t m_edge_count;
    size_t m_edge_type_count;
    size_t m_node_type_count;
    size_t m_node_feature_count;
    size_t m_edge_feature_count;

    // Infrastructure related information about graph.
    size_t m_partition_count;
    std::string m_path;
    std::string m_config_path;

    std::vector<std::vector<float>> m_partition_node_weights;
    std::vector<std::vector<float>> m_partition_edge_weights;
    std::vector<size_t> m_node_count_per_type;
    std::vector<size_t> m_edge_count_per_type;
};
} // namespace snark

#endif // SNARK_METADATA_H
