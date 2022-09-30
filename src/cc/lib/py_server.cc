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

std::string safe_convert(const char *buffer)
{
    if (buffer)
    {
        return std::string(buffer);
    }
    return std::string();
}

int32_t StartServer(PyServer *graph, size_t count, uint32_t *partitions, const char *filename, const char *host_name,
                    const char *ssl_key, const char *ssl_cert, const char *ssl_root,
                    const PyPartitionStorageType storage_type_, const char *config_path, bool enable_threadpool)
{
    snark::PartitionStorageType storage_type = static_cast<snark::PartitionStorageType>(storage_type_);
    graph->server = std::make_unique<snark::GRPCServer>(
        std::make_shared<snark::GraphEngineServiceImpl>(
            safe_convert(filename), std::vector<uint32_t>(partitions, partitions + count),
            static_cast<snark::PartitionStorageType>(storage_type), config_path, enable_threadpool),
        std::make_shared<snark::GraphSamplerServiceImpl>(safe_convert(filename),
                                                         std::set<size_t>(partitions, partitions + count)),
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
