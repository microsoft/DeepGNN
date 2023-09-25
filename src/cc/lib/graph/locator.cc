// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "locator.h"
#include <cstring>

namespace snark
{

FILE *open_file(std::filesystem::path s, const char *mode, std::shared_ptr<Logger> logger)
{
    if (!logger)
    {
        logger = std::make_shared<GLogger>();
    }
    FILE *f;
    if ((f = fopen(s.string().c_str(), mode)) == NULL)
    {
        logger->log_fatal("while opening file %s with: %s", s.string().c_str(), strerror(errno));
    }

    return f;
}

FILE *open_meta(std::filesystem::path path, std::string mode, std::shared_ptr<Logger> logger)
{
    return open_file(path / "meta.json", mode.c_str(), std::move(logger));
}

FILE *open_node_map(std::filesystem::path path, std::string suffix, std::shared_ptr<Logger> logger)
{
    return open_file(path / ("node_" + suffix + ".map"), "rb", std::move(logger));
}

FILE *open_node_index(std::filesystem::path path, std::string suffix, std::shared_ptr<Logger> logger)
{
    return open_file(path / ("node_" + suffix + ".index"), "rb", std::move(logger));
}

FILE *open_node_features_index(std::filesystem::path path, std::string suffix, std::shared_ptr<Logger> logger)
{
    return open_file(path / ("node_features_" + suffix + ".index"), "rb", std::move(logger));
}

FILE *open_node_features_data(std::filesystem::path path, std::string suffix, std::shared_ptr<Logger> logger)
{
    return open_file(path / ("node_features_" + suffix + ".data"), "rb", std::move(logger));
}

FILE *open_neighbor_index(std::filesystem::path path, std::string suffix, std::shared_ptr<Logger> logger)
{
    return open_file(path / ("neighbors_" + suffix + ".index"), "rb", std::move(logger));
}

FILE *open_edge_timestamps(std::filesystem::path path, std::string suffix, std::shared_ptr<Logger> logger)
{
    return open_file(path / ("edge_" + suffix + ".timestamp"), "rb", std::move(logger));
}

FILE *open_edge_index(std::filesystem::path path, std::string suffix, std::shared_ptr<Logger> logger)
{
    return open_file(path / ("edge_" + suffix + ".index"), "rb", std::move(logger));
}

FILE *open_edge_features_index(std::filesystem::path path, std::string suffix, std::shared_ptr<Logger> logger)
{
    return open_file(path / ("edge_features_" + suffix + ".index"), "rb", std::move(logger));
}

FILE *open_edge_features_data(std::filesystem::path path, std::string suffix, std::shared_ptr<Logger> logger)
{
    return open_file(path / ("edge_features_" + suffix + ".data"), "rb", std::move(logger));
}

FILE *open_edge_alias(std::filesystem::path path, size_t partition, Type type, std::shared_ptr<Logger> logger)
{
    return open_file(path / ("edge_" + std::to_string(type) + "_" + std::to_string(partition) + ".alias"), "rb",
                     std::move(logger));
}

FILE *open_node_alias(std::filesystem::path path, size_t partition, Type type, std::shared_ptr<Logger> logger)
{
    return open_file(path / ("node_" + std::to_string(type) + "_" + std::to_string(partition) + ".alias"), "rb",
                     std::move(logger));
}

void platform_fseek(FILE *f, int offset, int origin)
{
    // To work with large files on windows we need 64bit versions of fseek/ftell
#ifdef SNARK_PLATFORM_WINDOWS
    _fseeki64(f, offset, origin);
#else
    fseek(f, offset, origin);
#endif
}

size_t platform_ftell(FILE *f)
{
#ifdef SNARK_PLATFORM_WINDOWS
    return _ftelli64(f);
#else
    return ftell(f);
#endif
}

} // namespace snark
