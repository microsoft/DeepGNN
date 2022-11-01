// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "include/hdfs.h"
#include "src/cc/lib/graph/graph.h"
#include "src/cc/lib/graph/partition.h"
#include "src/cc/lib/graph/sampler.h"
#include "src/cc/tests/mocks.h"
#include "gtest/gtest.h"
#include <cstdio>
#include <filesystem>
#include <sstream>

void set_hdfs_env(std::filesystem::path temp_path, std::filesystem::path config_path)
{
    std::string hadoop_home_path = "external/hadoop/";

    int result = 0;
    result += setenv("HADOOP_HOME", hadoop_home_path.c_str(), 1);

    std::vector<std::string> classpath_exts = {
        "etc/hadoop:",
        "share/hadoop/common/lib/*:",
        "share/hadoop/common/*:",
        "share/hadoop/hdfs/lib/*:",
        "share/hadoop/hdfs/*:",
        "share/hadoop/tools/lib/hadoop-client-3.3.1.jar:",
    };

    std::string classpath = "";
    classpath.append(config_path.parent_path().string());
    classpath.append(":");
    for (auto &v : classpath_exts)
    {
        classpath.append(hadoop_home_path);
        classpath.append(v);
    }
    result += setenv("CLASSPATH", classpath.c_str(), 1);
}

TEST(HDFSTest, NodeTypes)
{
    auto temp_path = std::filesystem::temp_directory_path() / "hdfs_test_temp";
    std::string config_path = "src/cc/tests/core-site.xml";
    std::filesystem::create_directory(temp_path);
    TestGraph::MemoryGraph m;
    m.m_nodes.push_back(TestGraph::Node{.m_id = 0, .m_type = 0, .m_weight = 1.0f});
    m.m_nodes.push_back(TestGraph::Node{.m_id = 1, .m_type = 2, .m_weight = 1.0f});
    TestGraph::convert(temp_path, "0_0", std::move(m), 3);

    std::filesystem::path hdfs_path = std::string("file://") + temp_path.string();
    set_hdfs_env(temp_path, config_path);

    snark::Metadata metadata(hdfs_path.string(), config_path);
    snark::Graph g(std::move(metadata), std::vector<std::string>{hdfs_path.string()}, std::vector<uint32_t>{0},
                   snark::PartitionStorageType::memory);
    std::vector<snark::NodeId> nodes = {0, 1, 2};
    std::vector<snark::Type> output(3, -2);
    g.GetNodeType(std::span(nodes), std::span(output), -1);
    EXPECT_EQ(output, std::vector<snark::Type>({0, 2, -1}));
}

TEST(HDFSTest, NodeFeature)
{
    auto temp_path = std::filesystem::temp_directory_path() / "hdfs_test_temp";
    std::string config_path = "src/cc/tests/core-site.xml";
    std::filesystem::create_directory(temp_path);
    TestGraph::MemoryGraph m;
    std::vector<std::vector<float>> f1 = {std::vector<float>{1.0f, 2.0f, 3.0f}};
    std::vector<std::vector<float>> f2 = {std::vector<float>{5.0f, 6.0f, 7.0f}};
    m.m_nodes.push_back(TestGraph::Node{.m_id = 0, .m_type = 0, .m_weight = 1.0f, .m_float_features = f1});
    m.m_nodes.push_back(TestGraph::Node{.m_id = 1, .m_type = 1, .m_weight = 1.0f, .m_float_features = f2});
    TestGraph::convert(temp_path, "0_0", std::move(m), 3);

    std::filesystem::path hdfs_path = std::string("file://") + temp_path.string();
    set_hdfs_env(temp_path, config_path);

    snark::Metadata metadata(hdfs_path.string(), config_path);
    snark::Graph g(std::move(metadata), std::vector<std::string>{hdfs_path.string()}, std::vector<uint32_t>{0},
                   snark::PartitionStorageType::memory);
    std::vector<snark::NodeId> nodes = {0, 1};
    std::vector<uint8_t> output(4 * 3 * 2);
    std::vector<snark::FeatureMeta> features = {{0, 12}};
    g.GetNodeFeature(std::span(nodes), std::span(features), std::span(output));
    std::span res(reinterpret_cast<float *>(output.data()), output.size() / sizeof(float));
    EXPECT_EQ(std::vector<float>(std::begin(res), std::end(res)), std::vector<float>({1, 2, 3, 5, 6, 7}));
}

TEST(HDFSTest, NeighborSample)
{
    auto temp_path = std::filesystem::temp_directory_path() / "hdfs_test_temp";
    std::string config_path = "src/cc/tests/core-site.xml";
    std::filesystem::create_directory(temp_path);
    TestGraph::MemoryGraph m;
    m.m_nodes.push_back(TestGraph::Node{.m_id = 0,
                                        .m_type = 0,
                                        .m_weight = 1.0f,
                                        .m_neighbors{std::vector<TestGraph::NeighborRecord>{
                                            {1, 0, 1.0f}, {2, 0, 1.0f}, {3, 0, 1.0f}, {4, 0, 1.0f}, {5, 0, 1.0f}}}});
    m.m_nodes.push_back(TestGraph::Node{.m_id = 1, .m_type = 1, .m_weight = 1.0f});
    m.m_nodes.push_back(TestGraph::Node{
        .m_id = 2,
        .m_type = 1,
        .m_weight = 1.0f,
        .m_neighbors{std::vector<TestGraph::NeighborRecord>{{6, 0, 1.0f}, {7, 0, 1.0f}, {8, 0, 1.0f}}}});
    TestGraph::convert(temp_path, "0_0", std::move(m), 2);

    std::filesystem::path hdfs_path = std::string("file://") + temp_path.string();
    set_hdfs_env(temp_path, config_path);
    snark::Metadata metadata(hdfs_path.string(), config_path);
    snark::Graph g(std::move(metadata), std::vector<std::string>{hdfs_path.string()}, std::vector<uint32_t>{0},
                   snark::PartitionStorageType::memory);
    std::vector<snark::NodeId> nodes = {0, 2};
    std::vector<snark::Type> types = {0, 1};
    int count = 3;
    std::vector<snark::NodeId> neighbor_nodes(count * nodes.size(), -1);
    std::vector<snark::Type> neighbor_types(count * nodes.size(), -1);
    std::vector<float> neighbor_weights(count * nodes.size(), -1);
    std::vector<float> total_neighbor_weights(nodes.size());

    g.SampleNeighbor(42, std::span(nodes), std::span(types), count, std::span(neighbor_nodes),
                     std::span(neighbor_types), std::span(neighbor_weights), std::span(total_neighbor_weights), 0, 0,
                     -1);
    EXPECT_EQ(std::vector<snark::NodeId>({5, 1, 3, 7, 7, 8}), neighbor_nodes);
    EXPECT_EQ(std::vector<snark::Type>(6, 0), neighbor_types);
    EXPECT_EQ(std::vector<float>(6, 1), neighbor_weights);
}

TEST(HDFSTest, EdgeFeatureNoFeature)
{
    auto temp_path = std::filesystem::temp_directory_path() / "hdfs_test_temp";
    std::string config_path = "src/cc/tests/core-site.xml";
    std::filesystem::create_directory(temp_path);
    TestGraph::MemoryGraph m;
    std::vector<std::vector<float>> f1 = {std::vector<float>{1.0f, 2.0f, 3.0f}};
    std::vector<std::vector<float>> f2 = {std::vector<float>{11.0f, 12.0f}};
    m.m_nodes.push_back(TestGraph::Node{.m_id = 0, .m_type = 0, .m_weight = 1.0f, .m_float_features = f1});
    m.m_nodes.push_back(TestGraph::Node{.m_id = 1, .m_type = 1, .m_weight = 1.0f, .m_float_features = f2});
    m.m_nodes.push_back(TestGraph::Node{.m_id = 2, .m_type = 1, .m_weight = 1.0f, .m_float_features = f2});
    TestGraph::convert(temp_path, "0_0", std::move(m), 2);

    std::filesystem::path hdfs_path = std::string("file://") + temp_path.string();
    set_hdfs_env(temp_path, config_path);
    snark::Metadata metadata(hdfs_path.string(), config_path);
    snark::Graph g(std::move(metadata), std::vector<std::string>{hdfs_path.string()}, std::vector<uint32_t>{0},
                   snark::PartitionStorageType::memory);
    std::vector<snark::NodeId> nodes_src = {0};
    std::vector<snark::NodeId> nodes_dest = {1};
    std::vector<snark::Type> edge_types = {0};
    std::vector<uint8_t> large_output(4 * 3);
    std::vector<snark::FeatureMeta> features = {{1, 12}};
    g.GetEdgeFeature(std::span(nodes_src), std::span(nodes_dest), std::span(edge_types), std::span(features),
                     std::span(large_output));
    std::span large_res(reinterpret_cast<float *>(large_output.data()), large_output.size() / 4);
    EXPECT_EQ(std::vector<float>(std::begin(large_res), std::end(large_res)), std::vector<float>(3, 0.0f));
}

TEST(HDFSTest, HDFSLinkSplit)
{
    std::string data_path;
    std::string host;
    int port;

    // HDFS
    data_path = "";
    host = "default";
    port = 0;
    std::string hdfs_path = "hdfs://localhost:9000/path";
    parse_hdfs_path(hdfs_path, data_path, host, port);

    EXPECT_EQ(data_path, hdfs_path);
    EXPECT_EQ(host, "hdfs://localhost");
    EXPECT_EQ(port, 9000);

    // ADL
    data_path = "";
    host = "default";
    port = 0;
    std::string adl_path = "adl://name.azuredatalakestore.net/path";
    parse_hdfs_path(adl_path, data_path, host, port);

    EXPECT_EQ(data_path, adl_path);
    EXPECT_EQ(host, adl_path);
    EXPECT_EQ(port, 0);

    // FILE
    data_path = "";
    host = "default";
    port = 0;
    std::string file_path = "file:///path";
    parse_hdfs_path(file_path, data_path, host, port);

    EXPECT_EQ(data_path, file_path);
    EXPECT_EQ(host, file_path);
    EXPECT_EQ(port, 0);
}

TEST(HDFSTest, CoresitePath)
{
    std::string config_path_in;
    std::string config_path_out;

    config_path_in = "src/cc/tests/core-site.xml";
    config_path_out = "";
    std::getline(std::stringstream(getenv("CLASSPATH")), config_path_out, ':');
    EXPECT_EQ(config_path_out, "src/cc/tests");

    config_path_in = "src/cc/tests";
    config_path_out = "";
    std::getline(std::stringstream(getenv("CLASSPATH")), config_path_out, ':');
    EXPECT_EQ(config_path_out, config_path_in);
}

TEST(HDFSTest, IsHDFSTest)
{
    EXPECT_EQ(is_hdfs_path("hdfs://localhost:9000/path"), true);
    EXPECT_EQ(is_hdfs_path("adl://name.azuredatalakestore.net/path"), true);
    EXPECT_EQ(is_hdfs_path("file:///path"), true);
    EXPECT_EQ(is_hdfs_path("/path/to"), false);
    EXPECT_EQ(is_hdfs_path("path/to"), false);
}
