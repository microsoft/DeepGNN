/* Copyright (c) Microsoft Corporation.
   Licensed under the MIT License. */
{
    global:
        CreateLocalGraph;
        StartServer;
        CreateRemoteClient;
        DeleteClient;
        GetNodeFeature;
        GetNodeSparseFeature;
        GetNodeStringFeature;
        GetEdgeFeature;
        GetEdgeSparseFeature;
        GetEdgeStringFeature;
        StartServer;
        NeighborCount;
        GetNeighbors;
        WeightedSampleNeighbor;
        UniformSampleNeighbor;
        CreateWeightedNodeSampler;
        CreateUniformNodeSampler;
        CreateUniformNodeSamplerWithoutReplacement;
        SampleNodes;
        CreateWeightedEdgeSampler;
        CreateUniformEdgeSampler;
        CreateUniformEdgeSamplerWithoutReplacement;
        SampleEdges;
        DeleteSampler;
        ResetSampler;
        ResetGraph;
        ResetServer;
        RandomWalk;
        GetNodeType;
        HDFSMoveMeta;
    local: *;
};
