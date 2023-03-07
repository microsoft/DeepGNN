// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "metadata.h"
#include "locator.h"

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <glog/logging.h>
#include <glog/raw_logging.h>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

namespace snark
{

Metadata::Metadata(std::filesystem::path path, std::string config_path)
    : m_path(path.string()), m_config_path(config_path)
{
    if (is_hdfs_path(path))
#ifndef SNARK_PLATFORM_LINUX
        RAW_LOG_FATAL("HDFS streaming only supported on Linux!");
#else
    {
        auto full_path = path / "meta.txt";
        auto buffer = read_hdfs<char>(full_path.string(), m_config_path);

        path = std::filesystem::temp_directory_path();
        auto meta_write = open_meta(path, "w");
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

    std::ifstream f(path / "meta.txt");
    json meta = json::parse(f);

    m_version = meta["binary_data_version"];
    if (m_version < MINIMUM_SUPPORTED_VERSION)
    {
        RAW_LOG_FATAL("Unsupported version of binary data %zu. Minimum supported version is %zu. Please use latest "
                      "deepgnn package to convert data.",
                      m_version, MINIMUM_SUPPORTED_VERSION);
    }

    m_node_count = meta["node_count"];
    m_edge_count = meta["edge_count"];
    m_node_type_count = meta["node_type_num"];
    m_edge_type_count = meta["edge_type_num"];
    m_node_feature_count = meta["node_feature_num"];
    m_edge_feature_count = meta["edge_feature_num"];
    m_partition_count = meta["n_partitions"];

    m_partition_node_weights =
        std::vector<std::vector<float>>(m_partition_count, std::vector<float>(m_node_type_count, 0.0f));
    m_partition_edge_weights =
        std::vector<std::vector<float>>(m_partition_count, std::vector<float>(m_edge_type_count, 0.0f));

    uint32_t partition_num;
    for (size_t p = 0; p < m_partition_count; ++p)
    {
        partition_num = meta["partition_ids"][p];
        std::string m_node_type_count_str = "node_weight_";
        m_node_type_count_str += std::to_string(partition_num);
        std::string m_edge_type_count_str = "edge_weight_";
        m_edge_type_count_str += std::to_string(partition_num);

        for (size_t i = 0; i < m_node_type_count; ++i)
        {
            m_partition_node_weights[partition_num][i] = meta[m_node_type_count_str][i];
        }
        for (size_t i = 0; i < m_edge_type_count; ++i)
        {
            m_partition_edge_weights[partition_num][i] = meta[m_edge_type_count_str][i];
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
    json json_meta = {
        {"binary_data_version", m_version},
        {"node_count", m_node_count},
        {"edge_count", m_edge_count},
        {"node_type_num", m_node_type_count},
        {"edge_type_num", m_edge_type_count},
        {"node_feature_num", m_node_feature_count},
        {"edge_feature_num", m_edge_feature_count},
        {"n_partitions", m_partition_count},
    };

    std::vector<size_t> partition_ids;

    for (size_t p = 0; p < m_partition_count; ++p)
    {
        partition_ids.push_back(p);

        std::string m_node_type_count_str = "node_weight_";
        m_node_type_count_str += std::to_string(p);
        std::string m_edge_type_count_str = "edge_weight_";
        m_edge_type_count_str += std::to_string(p);

        json_meta[m_node_type_count_str] = m_partition_node_weights[p];
        json_meta[m_edge_type_count_str] = m_partition_edge_weights[p];
    }

    json_meta["partition_ids"] = partition_ids;

    json_meta["node_count_per_type"] = m_node_count_per_type;
    json_meta["edge_count_per_type"] = m_edge_count_per_type;

    std::ofstream meta(path / "meta.txt");
    meta << json_meta << std::endl;
    meta.close();
}

} // namespace snark
