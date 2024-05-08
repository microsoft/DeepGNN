// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#ifndef SNARK_RESERVOIR_H
#define SNARK_RESERVOIR_H

#include <functional>
#include <span>

#include "boost/random/uniform_real_distribution.hpp"

#include "xoroshiro.h"

namespace snark
{

// Implementation of an optimal algorithm for reservoir sampling:
// https://en.wikipedia.org/wiki/Reservoir_sampling#Optimal:_Algorithm_L
class AlgorithmL
{
  public:
    // `k` is the size of reservoir, short name to be consistent with reference.
    // gen is the random generator to use for sampling and support for setting seeds in clients.
    AlgorithmL(size_t k, snark::Xoroshiro128PlusGenerator &gen);

    // add n elements to the reservoir. Every time an element is selected to put in a reservoir,
    // the update function is called. We use a callback approach, because we usually need to fetch
    // elements from multiple sources (edge type, destination, timestamps). Arguments passed to the callback
    // are (pick, offset). Pick is the index of element in the reservoir to be replaced in the range [0; k).
    // Offset is the offset in the stream in the range [0, n).
    // Method might be called multiple times which will result in merging multiple streams into one.
    void add(size_t n, std::function<void(size_t, size_t)> update);

  private:
    size_t m_k;
    float m_W;
    size_t m_next;
    size_t m_seen;
    snark::Xoroshiro128PlusGenerator &m_gen;
    boost::random::uniform_real_distribution<float> m_dist;
};

// Following paper "Reservoir-based Random Sampling with Replacement from Data Stream" by BH Park · 2004
class WithReplacement
{
  public:
    WithReplacement(size_t k, snark::Xoroshiro128PlusGenerator &gen);

    void add(size_t n, std::function<void(size_t, size_t)> update);

    void reset();

  private:
    size_t m_seen;
    size_t m_k;
    snark::Xoroshiro128PlusGenerator &m_gen;
    boost::random::uniform_real_distribution<float> m_dist;
};

// Used for merging multiple sampled neighbors lists into one. We can't use WithReplacement directly,
// because we need to consider intervals with smaller than k elements: if we have two lists
// of equal sizes 10, we can't use bernulli trials to merge them into one list of size 15,
// because we need to backfill first and then sample from the merged reservoir, but with updated weights.
class WithoutReplacementMerge
{
  public:
    WithoutReplacementMerge(size_t k, snark::Xoroshiro128PlusGenerator &gen);

    // w in this case is the weight of the interval, not the number of elements as in classes above.
    void add(size_t w, std::function<void(size_t, size_t)> update);

    void reset();

  private:
    size_t m_seen;
    size_t m_k;
    snark::Xoroshiro128PlusGenerator &m_gen;
    boost::random::uniform_real_distribution<float> m_dist;
};

} // namespace snark

#endif
