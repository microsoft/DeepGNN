#include <cassert>
#include <cmath>

#include "reservoir.h"
#include <algorithm>
#include <numeric>

namespace snark
{

AlgorithmL::AlgorithmL(size_t k, snark::Xoroshiro128PlusGenerator &gen)
    : m_k(k), m_W{0}, m_next(0), m_seen(0), m_gen(gen),
      m_dist(boost::random::uniform_real_distribution<float>(0.0f, 1.0f))
{
    assert(k > 0);
}

void AlgorithmL::add(size_t n, std::function<void(size_t, size_t)> update)
{
    size_t i = 0;
    for (; m_seen < m_k && i < n; ++m_seen, ++i, ++m_next)
    {
        update(m_seen, i);
    }

    size_t left = n - i;
    // Check if we can postpone random number generation.
    if (left == 0)
    {
        return;
    }

    if (m_seen == m_k)
    {
        m_W = std::exp(std::log(m_dist(m_gen)) / m_k);
        m_next += std::floor(std::log(m_dist(m_gen)) / std::log(1 - m_W)) + 1;
    }

    while (m_next <= m_seen + left) // First time we've seen more than m_k elements
    {
        const size_t range_offset = m_seen + left - m_next + i;
        const size_t new_left = n - range_offset;
        m_seen += left - new_left;
        left = new_left;
        size_t pick = m_dist(m_gen) * m_k;
        update(pick, range_offset);

        m_W *= std::exp(std::log(m_dist(m_gen)) / m_k);
        m_next += std::floor(std::log(m_dist(m_gen)) / std::log(1 - m_W)) + 1;
    }

    m_seen += left;
}

WithReplacement::WithReplacement(size_t k, snark::Xoroshiro128PlusGenerator &gen)
    : m_seen(0), m_k(k), m_gen(gen), m_dist(boost::random::uniform_real_distribution<float>(0.0f, 1.0f))
{
    assert(k > 0);
}

void WithReplacement::add(size_t n, std::function<void(size_t, size_t)> update)
{
    m_seen += n;
    const float rate = float(n) / m_seen;
    for (size_t i = 0; i < m_k; ++i)
    {
        if (rate == 1.0f || m_dist(m_gen) < rate)
        {
            update(i, size_t(n * m_dist(m_gen)));
        }
    }
}

void WithReplacement::reset()
{
    m_seen = 0;
}

WithoutReplacementMerge::WithoutReplacementMerge(size_t k, snark::Xoroshiro128PlusGenerator &gen)
    : m_seen(0), m_k(k), m_gen(gen), m_dist(boost::random::uniform_real_distribution<float>(0.0f, 1.0f))
{
    assert(k > 0);
}

void WithoutReplacementMerge::add(size_t n, std::function<void(size_t, size_t)> update)
{
    if (m_seen + n <= m_k)
    {
        for (size_t i = 0; i < n; ++i)
        {
            update(m_seen + i, i);
        }
        m_seen += n;
        return;
    }

    size_t left_count = 0;
    size_t right_count = 0;
    size_t left_weight = m_seen;
    size_t right_weight = n;

    // To merge two reservoirs of size m_seen and n, we need to sample m_k elements with variable
    // probabilities proportional to the weights of remainnig elements in the reservoirs.
    for (size_t i = 0; i < m_k; ++i)
    {
        if (m_dist(m_gen) < float(left_weight) / (left_weight + right_weight))
        {
            ++left_count;
            --left_weight;
        }
        else
        {
            ++right_count;
            --right_weight;
        }
    }

    // Randomly shuffle the indices of the left and right reservoirs to determine final indices.
    size_t left_size = m_seen;
    size_t right_size = std::min(n, m_k);
    std::vector<size_t> left_indices(left_size);
    std::vector<size_t> right_indices(right_size);
    std::iota(std::begin(left_indices), std::end(left_indices), 0);
    std::iota(std::begin(right_indices), std::end(right_indices), 0);
    std::shuffle(std::begin(left_indices), std::end(left_indices), m_gen);
    std::shuffle(std::begin(right_indices), std::end(right_indices), m_gen);

    // Pick first left_count and right_count elements from the shuffled arrays to avoid repetitive subsampling
    // of the same element.
    std::sort(std::begin(left_indices), std::begin(left_indices) + left_count);
    size_t left_index = 0;
    size_t right_index = 0;
    for (size_t i = 0; i < m_k; ++i)
    {
        if (left_index < left_count && left_indices[left_index] == i)
        {
            ++left_index;
        }
        else
        {
            update(i, right_indices[right_index]);
            ++right_index;
        }
    }
}

void WithoutReplacementMerge::reset()
{
    m_seen = 0;
}

} // namespace snark
