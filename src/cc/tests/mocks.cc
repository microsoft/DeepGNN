// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "src/cc/tests/mocks.h"
#include "src/cc/lib/graph/metadata.h"

#include <fstream>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

namespace TestGraph
{
void save_edge_timestamps(std::ofstream &out,
                          const std::vector<std::pair<snark::Timestamp, snark::Timestamp>> &timestamps,
                          snark::Timestamp watermark)
{
    out.write(reinterpret_cast<const char *>(&watermark), sizeof(snark::Timestamp));
    for (const auto &ts : timestamps)
    {
        out.write(reinterpret_cast<const char *>(&ts.first), sizeof(snark::Timestamp));
        out.write(reinterpret_cast<const char *>(&ts.second), sizeof(snark::Timestamp));
    }
}
snark::Partition convert(std::filesystem::path path, std::string suffix, MemoryGraph t, size_t node_types)
{
    std::vector<NeighborRecord> edge_index;
    std::vector<uint64_t> nb_index;
    std::vector<float> node_type_weights(node_types, 0.0f);
    std::vector<size_t> node_type_counts(node_types);
    std::vector<std::vector<std::vector<float>>> edge_features;
    int64_t counter = 0;
    {
        std::fstream node_map(path / ("node_" + suffix + ".map"), std::ios_base::binary | std::ios_base::out);
        for (auto n : t.m_nodes)
        {
            node_map.write(reinterpret_cast<const char *>(&n.m_id), sizeof(snark::NodeId));
            node_map.write(reinterpret_cast<const char *>(&counter), sizeof(int64_t));
            node_map.write(reinterpret_cast<const char *>(&n.m_type), sizeof(snark::Type));
            if (n.m_type >= 0)
            {
                node_type_weights[n.m_type] += n.m_weight;
                ++node_type_counts[n.m_type];
            }
            ++counter;
            nb_index.push_back(edge_index.size());
            for (auto &nb : n.m_neighbors)
            {
                edge_index.emplace_back(nb);
            }
            for (auto &features : n.m_edge_features)
            {
                edge_features.emplace_back(features);
            }
        }
        node_map.close();
    }
    {
        nb_index.push_back(edge_index.size());
        std::ofstream nb_out(path / ("neighbors_" + suffix + ".index"), std::ios_base::binary | std::ios_base::out);
        for (auto i : nb_index)
        {
            nb_out.write(reinterpret_cast<const char *>(&i), sizeof(uint64_t));
        }
        nb_out.close();
    }
    {
        std::ofstream edge_index_out(path / ("edge_" + suffix + ".index"), std::ios_base::binary | std::ios_base::out);
        std::ofstream edge_feature_index_out(path / ("edge_features_" + suffix + ".index"),
                                             std::ios_base::binary | std::ios_base::out);
        std::ofstream edge_feature_data_out(path / ("edge_features_" + suffix + ".data"),
                                            std::ios_base::binary | std::ios_base::out);

        for (size_t edge_pos = 0; edge_pos < edge_index.size(); ++edge_pos)
        {
            auto &e = edge_index[edge_pos];
            auto dst = std::get<0>(e);
            edge_index_out.write(reinterpret_cast<const char *>(&dst), sizeof(snark::NodeId));
            uint64_t feature_offset = edge_feature_index_out.tellp();
            edge_index_out.write(reinterpret_cast<const char *>(&feature_offset), sizeof(uint64_t));
            for (size_t feature_pos = 0; !edge_features.empty() && feature_pos < edge_features[edge_pos].size();
                 ++feature_pos)
            {
                uint64_t feature_offset = edge_feature_data_out.tellp();
                edge_feature_index_out.write(reinterpret_cast<const char *>(&feature_offset), sizeof(uint64_t));
                auto &fv = edge_features[edge_pos][feature_pos];
                edge_feature_data_out.write(reinterpret_cast<const char *>(fv.data()), fv.size() * sizeof(float));
            }
            auto type = std::get<1>(e);
            edge_index_out.write(reinterpret_cast<const char *>(&type), sizeof(snark::Type));
            auto weight = std::get<2>(e);
            edge_index_out.write(reinterpret_cast<const char *>(&weight), sizeof(float));
        }
        int64_t dst = -1;
        edge_index_out.write(reinterpret_cast<const char *>(&dst), sizeof(snark::NodeId));

        uint64_t feature_data_offset = edge_feature_data_out.tellp();
        edge_feature_index_out.write(reinterpret_cast<const char *>(&feature_data_offset), sizeof(uint64_t));
        uint64_t feature_index_offset = edge_feature_index_out.tellp();
        edge_index_out.write(reinterpret_cast<const char *>(&feature_index_offset), sizeof(uint64_t));
        int32_t type = -1;
        edge_index_out.write(reinterpret_cast<const char *>(&type), sizeof(snark::Type));
        float weight = 1;
        edge_index_out.write(reinterpret_cast<const char *>(&weight), sizeof(float));

        edge_feature_data_out.close();
        edge_feature_index_out.close();
        edge_index_out.close();
    }

    if (!t.m_edge_timestamps.empty())
    {
        std::ofstream edge_timestamps_out(path / ("edge_" + suffix + ".timestamp"),
                                          std::ios_base::binary | std::ios_base::out);
        save_edge_timestamps(edge_timestamps_out, t.m_edge_timestamps, t.m_watermark);
    }

    {
        std::string version_str = "v";
        version_str += std::to_string(snark::MINIMUM_SUPPORTED_VERSION);

        json json_meta = {
            {"binary_data_version", version_str},
            {"node_count", counter},
            {"edge_count", nb_index.size()},
            {"node_type_count", node_types},
            {"edge_type_count", 1},
            {"node_feature_count", 1},
            {"edge_feature_count", (edge_features.empty() ? 0 : 1)},
            {"watermark", t.m_watermark},
        };

        json_meta["partitions"] = {{"0", {{"node_weight", node_type_weights}, {"edge_weight", {1}}}}};
        json_meta["node_count_per_type"] = node_type_counts;
        json_meta["edge_count_per_type"] = {edge_index.size()};

        std::ofstream meta(path / "meta.json");
        meta << json_meta << std::endl;
        meta.close();
    }
    {
        std::ofstream node_index_out(path / ("node_" + suffix + ".index"), std::ios_base::binary | std::ios_base::out);

        std::ofstream feature_index_out(path / ("node_features_" + suffix + ".index"),
                                        std::ios_base::binary | std::ios_base::out);

        std::ofstream feature_data_out(path / ("node_features_" + suffix + ".data"),
                                       std::ios_base::binary | std::ios_base::out);
        for (auto &node : t.m_nodes)
        {
            auto node_pos = (uint64_t)feature_index_out.tellp() / sizeof(uint64_t);
            node_index_out.write(reinterpret_cast<const char *>(&node_pos), sizeof(uint64_t));
            for (auto &f : node.m_float_features)
            {
                auto pos = uint64_t(feature_data_out.tellp());
                feature_index_out.write(reinterpret_cast<const char *>(&pos), sizeof(uint64_t));
                feature_data_out.write(reinterpret_cast<const char *>(f.data()), sizeof(float) * f.size());
            }
        }
        auto node_pos = (uint64_t)feature_index_out.tellp() / sizeof(uint64_t);
        node_index_out.write(reinterpret_cast<const char *>(&node_pos), sizeof(uint64_t));
        node_index_out.close();

        auto pos = (uint64_t)feature_data_out.tellp();
        feature_index_out.write(reinterpret_cast<const char *>(&pos), sizeof(uint64_t));

        feature_index_out.close();
        feature_data_out.close();
    }
    {
        std::ofstream node_index_out(path / ("node_" + suffix + ".index"), std::ios_base::binary | std::ios_base::out);

        std::ofstream feature_index_out(path / ("node_features_" + suffix + ".index"),
                                        std::ios_base::binary | std::ios_base::out);

        std::ofstream feature_data_out(path / ("node_features_" + suffix + ".data"),
                                       std::ios_base::binary | std::ios_base::out);
        for (auto &node : t.m_nodes)
        {
            auto node_pos = uint64_t(feature_index_out.tellp()) / sizeof(uint64_t);
            node_index_out.write(reinterpret_cast<const char *>(&node_pos), sizeof(uint64_t));
            for (auto &f : node.m_float_features)
            {
                auto pos = uint64_t(feature_data_out.tellp());
                feature_index_out.write(reinterpret_cast<const char *>(&pos), sizeof(uint64_t));
                feature_data_out.write(reinterpret_cast<const char *>(f.data()), sizeof(float) * f.size());
            }
        }
        auto node_pos = (uint64_t)feature_index_out.tellp() / sizeof(uint64_t);
        node_index_out.write(reinterpret_cast<const char *>(&node_pos), sizeof(uint64_t));
        node_index_out.close();

        auto pos = (uint64_t)feature_data_out.tellp();
        feature_index_out.write(reinterpret_cast<const char *>(&pos), sizeof(uint64_t));

        feature_index_out.close();
        feature_data_out.close();
    }

    return snark::Partition(snark::Metadata(path.string(), path.string()), path, suffix,
                            snark::PartitionStorageType::memory);
}

std::vector<float> serialize_temporal_features(std::vector<snark::Timestamp> timestamps,
                                               std::vector<std::vector<float>> features)
{
    assert(timestamps.size() == features.size());
    size_t total_size = std::accumulate(std::begin(features), std::end(features), size_t(0),
                                        [](size_t curr, const std::vector<float> &fv) { return curr + fv.size() * 4; });
    size_t metadata_size = timestamps.size() * 2 + 1;
    total_size += 2 * metadata_size + 1; // floats are 32 bits, ts is 64.
    std::vector<float> output(total_size);
    auto interval_count = reinterpret_cast<uint32_t *>(output.data());
    *interval_count = uint32_t(timestamps.size());
    auto metadata = std::span<int64_t>(reinterpret_cast<int64_t *>(output.data() + 1), metadata_size);
    size_t curr_pos = 0;
    for (auto ts : timestamps)
    {
        metadata[curr_pos] = int64_t(ts);
        ++curr_pos;
    }

    // Final offset to simplify distance calculation.
    size_t data_offset = (4 + 8 * (2 * timestamps.size() + 1)) / 4;
    auto data_start = std::begin(output) + data_offset;
    for (const auto &fv : features)
    {
        metadata[curr_pos] = int64_t(data_offset * sizeof(float));
        data_start = std::copy(std::begin(fv), std::end(fv), data_start);
        ++curr_pos;
        data_offset += fv.size();
    }

    metadata[curr_pos] = int64_t(data_offset * sizeof(float));
    return output;
}
} // namespace TestGraph
