// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "py_graph.h"

#include <cstdint>
#include <memory>
#include <set>
#include <span>
#include <string>
#include <vector>

#include "distributed/server.h"

namespace deep_graph
{
namespace python
{

namespace
{
std::string safe_convert(const char *buffer)
{
    if (buffer)
    {
        return std::string(buffer);
    }
    return std::string();
}
} // namespace

int32_t StartServer(PyServer *graph, const char *meta_location, size_t count, uint32_t *partition_indices,
                    const char **partition_locations, const char *host_name, const char *ssl_key, const char *ssl_cert,
                    const char *ssl_root, const PyPartitionStorageType storage_type_, const char *config_path)
{
    snark::PartitionStorageType storage_type = static_cast<snark::PartitionStorageType>(storage_type_);
    snark::Metadata metadata(safe_convert(meta_location), safe_convert(config_path));
    std::vector<std::string> partition_paths;
    partition_paths.reserve(count);
    for (size_t i = 0; i < count; ++i)
    {
        partition_paths.emplace_back(safe_convert(partition_locations[i]));
    }
    graph->server = std::make_unique<snark::GRPCServer>(
        std::make_shared<snark::GraphEngineServiceImpl>(
            metadata, partition_paths, std::vector<uint32_t>(partition_indices, partition_indices + count),
            static_cast<snark::PartitionStorageType>(storage_type)),
        std::make_shared<snark::GraphSamplerServiceImpl>(
            metadata, partition_paths, std::vector<size_t>(partition_indices, partition_indices + count)),
        safe_convert(host_name), safe_convert(ssl_key), safe_convert(ssl_cert), safe_convert(ssl_root));
    return 0;
}

int32_t ResetServer(PyServer *py_graph)
{
    py_graph->server.reset();
    return 0;
}

} // namespace python
} // namespace deep_graph
