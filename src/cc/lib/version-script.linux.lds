/* Copyright (c) Microsoft Corporation.
   Licensed under the MIT License. */
{
    global:
        CreateLocalGraph;
        StartServer;
        CreateRemoteClient;
        GetNodeFeature;
        GetNodeSparseFeature;
        GetEdgeFeature;
        GetEdgeSparseFeature;
        StartServer;
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
        ResetSampler;
        ResetGraph;
        ResetServer;
        RandomWalk;
        GetNodeType;
        HDFSMoveMeta;
    local: *;
};
