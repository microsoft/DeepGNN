// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "metadata.h"
#include "locator.h"
#include "logger.h"

#include <cinttypes>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

namespace snark
{

Metadata::Metadata(std::filesystem::path path, std::string config_path, std::shared_ptr<Logger> logger,
                   bool skip_feature_loading)
    : m_version(MINIMUM_SUPPORTED_VERSION), m_path(path.string()), m_config_path(config_path), m_watermark(-1)
{

    if (!logger)
    {
        logger = std::make_shared<GLogger>();
    }
    if (is_hdfs_path(path))
#ifndef SNARK_PLATFORM_LINUX
        logger->log_fatal("HDFS streaming only supported on Linux!");
#else
    {
        auto full_path = path / "meta.json";
        auto buffer = read_hdfs<char>(full_path.string(), m_config_path);

        path = std::filesystem::temp_directory_path();
        auto meta_write = open_meta(path, "w", logger);
        for (uint64_t i = 0; i < buffer.size(); ++i)
        {
            if (fprintf(meta_write, "%c", buffer[i]) <= 0)
            {
                exit(errno);
            }
        }
        fclose(meta_write);
    }
#endif

    std::ifstream f(path / "meta.json");
    json meta = json::parse(f);

    std::string version_full = meta["binary_data_version"];
    version_full.erase(0, 1);
    m_version = stoi(version_full);
    if (m_version < MINIMUM_SUPPORTED_VERSION)
    {
        logger->log_fatal("Unsupported version of binary data %zu. Minimum supported version is %zu. Please use latest "
                          "deepgnn package to convert data.",
                          m_version, MINIMUM_SUPPORTED_VERSION);
    }

    m_node_count = meta["node_count"];
    m_edge_count = meta["edge_count"];
    m_node_type_count = meta["node_type_count"];
    m_edge_type_count = meta["edge_type_count"];
    m_node_feature_count = meta["node_feature_count"];
    m_edge_feature_count = meta["edge_feature_count"];
    m_partition_count = meta["partitions"].size();
    m_watermark = meta["watermark"];
    // Skip feature loading if requested. This is useful when the feature loading is done in a separate feature store.
    if (skip_feature_loading == true)
    {
        m_node_feature_count = 0;
        m_edge_feature_count = 0;
    }

    m_partition_node_weights =
        std::vector<std::vector<float>>(m_partition_count, std::vector<float>(m_node_type_count, 0.0f));
    m_partition_edge_weights =
        std::vector<std::vector<float>>(m_partition_count, std::vector<float>(m_edge_type_count, 0.0f));

    for (size_t p = 0; p < m_partition_count; ++p)
    {
        for (size_t i = 0; i < m_node_type_count; ++i)
        {
            m_partition_node_weights[p][i] = meta["partitions"][std::to_string(p)]["node_weight"][i];
        }
        for (size_t i = 0; i < m_edge_type_count; ++i)
        {
            m_partition_edge_weights[p][i] = meta["partitions"][std::to_string(p)]["edge_weight"][i];
        }
    }

    m_node_count_per_type.resize(m_node_type_count);
    for (size_t i = 0; i < m_node_type_count; ++i)
    {
        m_node_count_per_type[i] = meta["node_count_per_type"][i];
    }
    m_edge_count_per_type.resize(m_edge_type_count);
    for (size_t i = 0; i < m_edge_type_count; ++i)
    {
        m_edge_count_per_type[i] = meta["edge_count_per_type"][i];
    }
}

void Metadata::Write(std::filesystem::path path) const
{
    std::string version_str = "v";
    version_str += std::to_string(m_version);

    json json_meta = {
        {"binary_data_version", version_str},
        {"node_count", m_node_count},
        {"edge_count", m_edge_count},
        {"node_type_count", m_node_type_count},
        {"edge_type_count", m_edge_type_count},
        {"node_feature_count", m_node_feature_count},
        {"edge_feature_count", m_edge_feature_count},
        {"watermark", m_watermark},
    };

    json_meta["partitions"] = {{"0", {{"node_weight", {0}}}}};
    for (size_t p = 0; p < m_partition_count; ++p)
    {
        json_meta["partitions"][std::to_string(p)] = {{"node_weight", {0}}};
        json_meta["partitions"][std::to_string(p)]["node_weight"] = m_partition_node_weights[p];
        json_meta["partitions"][std::to_string(p)]["edge_weight"] = m_partition_edge_weights[p];
    }

    json_meta["node_count_per_type"] = m_node_count_per_type;
    json_meta["edge_count_per_type"] = m_edge_count_per_type;

    std::ofstream meta(path / "meta.json");
    meta << json_meta;
    meta.close();
}

} // namespace snark
