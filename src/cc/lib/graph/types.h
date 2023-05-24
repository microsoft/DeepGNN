// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#ifndef SNARK_TYPES_H
#define SNARK_TYPES_H
#include <cstdint>
#include <utility>

namespace snark
{

using NodeId = int64_t;
using Type = int32_t;
using FeatureId = int32_t;
using FeatureSize = uint32_t;
using Timestamp = int64_t;
using FeatureMeta = std::pair<FeatureId, FeatureSize>;

const int32_t PLACEHOLDER_NODE_TYPE = -1;

// Enum ordering should match PyPartitionStorageType in py_graph.h.
enum PartitionStorageType
{
    memory,
    disk,
};

} // namespace snark
#endif
