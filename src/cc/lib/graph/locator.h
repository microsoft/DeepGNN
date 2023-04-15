// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#ifndef SNARK_LOCATOR_H
#define SNARK_LOCATOR_H

#include <filesystem>
#include <string>

#include "hdfs_wrap.h"
#include "types.h"

namespace snark
{
FILE *open_file(std::filesystem::path s, const char *mode);
FILE *open_meta(std::filesystem::path path, std::string mode);
FILE *open_node_map(std::filesystem::path path, std::string suffix);
FILE *open_node_index(std::filesystem::path path, std::string suffix);
FILE *open_node_features_index(std::filesystem::path path, std::string suffix);
FILE *open_node_features_data(std::filesystem::path path, std::string suffix);
FILE *open_neighbor_index(std::filesystem::path path, std::string suffix);
FILE *open_edge_timestamps(std::filesystem::path path, std::string suffix);
FILE *open_edge_index(std::filesystem::path path, std::string suffix);
FILE *open_edge_features_index(std::filesystem::path path, std::string suffix);
FILE *open_edge_features_data(std::filesystem::path path, std::string suffix);
FILE *open_edge_alias(std::filesystem::path path, size_t partition, Type type);
FILE *open_node_alias(std::filesystem::path path, size_t partition, Type type);

void platform_fseek(FILE *f, int offset, int origin);
size_t platform_ftell(FILE *f);
}; // namespace snark

#endif // SNARK_LOCATOR_H
