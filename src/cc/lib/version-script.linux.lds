/* Copyright (c) Microsoft Corporation.
   Licensed under the MIT License. */
{
    global:
        CreateLocalGraph;
        StartServer;
        CreateRemoteClient;
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
        PPRSampleNeighbor;
        CreateWeightedNodeSampler;
        CreateUniformNodeSampler;
        CreateUniformNodeSamplerWithoutReplacement;
        SampleNodes;
        CreateWeightedEdgeSampler;
        CreateUniformEdgeSampler;
        CreateUniformEdgeSamplerWithoutReplacement;
        SampleEdges;
        ResetSampler;
        ResetGraph;
        ResetServer;
        RandomWalk;
        GetNodeType;
        HDFSMoveMeta;
    local: *;
};
