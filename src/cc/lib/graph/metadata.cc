// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "metadata.h"
#include "locator.h"

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <glog/logging.h>
#include <glog/raw_logging.h>

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

    auto meta = open_meta(std::move(path), "rt");
    if (fscanf(meta, "v%zu\n", &m_version) <= 0)
    {
        RAW_LOG_ERROR(
            "Failed to read binary data version from meta file. Please use latest deepgnn package to convert data.");
        exit(errno);
    }

    if (m_version < MINIMUM_SUPPORTED_VERSION)
    {
        RAW_LOG_FATAL("Unsupported version of binary data %zu. Minimum supported version is %zu. Please use latest "
                      "deepgnn package to convert data.",
                      m_version, MINIMUM_SUPPORTED_VERSION);
    }

    if (fscanf(meta, "%zu\n", &m_node_count) <= 0)
    {
        exit(errno);
    }
    if (fscanf(meta, "%zu\n", &m_edge_count) <= 0)
    {
        exit(errno);
    }
    if (fscanf(meta, "%zu\n", &m_node_type_count) <= 0)
    {
        exit(errno);
    }
    if (fscanf(meta, "%zu\n", &m_edge_type_count) <= 0)
    {
        exit(errno);
    }
    if (fscanf(meta, "%zu\n", &m_node_feature_count) <= 0)
    {
        exit(errno);
    }
    if (fscanf(meta, "%zu\n", &m_edge_feature_count) <= 0)
    {
        exit(errno);
    }

    if (fscanf(meta, "%zu\n", &m_partition_count) <= 0)
    {
        exit(errno);
    }

    m_partition_node_weights =
        std::vector<std::vector<float>>(m_partition_count, std::vector<float>(m_node_type_count, 0.0f));
    m_partition_edge_weights =
        std::vector<std::vector<float>>(m_partition_count, std::vector<float>(m_edge_type_count, 0.0f));

    float edge_weight, node_weight;
    uint32_t partition_num;
    for (size_t p = 0; p < m_partition_count; ++p)
    {
        if (fscanf(meta, "%ul\n", &partition_num) <= 0)
        {
            exit(errno);
        }
        for (size_t i = 0; i < m_node_type_count; ++i)
        {
            if (fscanf(meta, "%f\n", &node_weight) <= 0)
            {
                exit(errno);
            }
            m_partition_node_weights[partition_num][i] = node_weight;
        }
        for (size_t i = 0; i < m_edge_type_count; ++i)
        {
            if (fscanf(meta, "%f\n", &edge_weight) <= 0)
            {
                exit(errno);
            }
            m_partition_edge_weights[partition_num][i] = edge_weight;
        }
    }
    size_t count;
    m_node_count_per_type.resize(m_node_type_count);
    for (size_t i = 0; i < m_node_type_count; ++i)
    {
        if (fscanf(meta, "%zu\n", &count) <= 0)
        {
            exit(errno);
        }
        m_node_count_per_type[i] = count;
    }
    m_edge_count_per_type.resize(m_edge_type_count);
    for (size_t i = 0; i < m_edge_type_count; ++i)
    {
        if (fscanf(meta, "%zu\n", &count) <= 0)
        {
            exit(errno);
        }
        m_edge_count_per_type[i] = count;
    }
    fclose(meta);
}

void Metadata::Write(std::filesystem::path path) const
{
    auto meta = open_meta(std::move(path), "w+");
    if (fprintf(meta, "v%zu\n", m_version) <= 0)
    {
        exit(errno);
    }
    if (fprintf(meta, "%zu\n", m_node_count) <= 0)
    {
        exit(errno);
    }
    if (fprintf(meta, "%zu\n", m_edge_count) <= 0)
    {
        exit(errno);
    }
    if (fprintf(meta, "%zu\n", m_node_type_count) <= 0)
    {
        exit(errno);
    }
    if (fprintf(meta, "%zu\n", m_edge_type_count) <= 0)
    {
        exit(errno);
    }
    if (fprintf(meta, "%zu\n", m_node_feature_count) <= 0)
    {
        exit(errno);
    }
    if (fprintf(meta, "%zu\n", m_edge_feature_count) <= 0)
    {
        exit(errno);
    }

    if (fprintf(meta, "%zu\n", m_partition_count) <= 0)
    {
        exit(errno);
    }

    for (size_t p = 0; p < m_partition_count; ++p)
    {
        if (fprintf(meta, "%zu\n", p) <= 0)
        {
            exit(errno);
        }
        for (size_t i = 0; i < m_node_type_count; ++i)
        {
            if (fprintf(meta, "%f\n", m_partition_node_weights[p][i]) <= 0)
            {
                exit(errno);
            }
        }
        for (size_t i = 0; i < m_edge_type_count; ++i)
        {
            if (fprintf(meta, "%f\n", m_partition_edge_weights[p][i]) <= 0)
            {
                exit(errno);
            }
        }
    }
    for (size_t i = 0; i < m_node_type_count; ++i)
    {
        if (fprintf(meta, "%zu\n", m_node_count_per_type[i]) <= 0)
        {
            exit(errno);
        }
    }
    for (size_t i = 0; i < m_edge_type_count; ++i)
    {
        if (fprintf(meta, "%zu\n", m_edge_count_per_type[i]) <= 0)
        {
            exit(errno);
        }
    }
    fclose(meta);
}

} // namespace snark
