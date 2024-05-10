#include <algorithm>
#include <cassert>
#include <cmath>
#include <numeric>
#include <vector>

#include "reservoir.h"

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

void WithReplacement::add(size_t w, std::function<void(size_t, size_t)> update)
{
    m_seen += w;
    const float rate = float(w) / m_seen;
    for (size_t i = 0; i < m_k; ++i)
    {
        if (rate == 1.0f || m_dist(m_gen) < rate)
        {
            update(i, size_t(w * m_dist(m_gen)));
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

namespace
{
// Use custom sample, because std::sample is not guaranteed to be deterministic across platforms.
using It = std::vector<size_t>::iterator;
void sample_n(It first, It last, size_t n, snark::Xoroshiro128PlusGenerator &gen)
{
    size_t total_elements = size_t(last - first);
    assert(total_elements > 0);
    assert(n <= total_elements && "Sample size is larger than the range");
    // There is no need to swap the last element with itself.
    total_elements -= 1;
    for (size_t i = 0; i < total_elements; ++i)
    {
        std::uniform_int_distribution<size_t> dist(i + 1, total_elements);
        std::swap(first[i], first[dist(gen)]);
    }
}
} // namespace

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
    if (left_count > 0)
    {
        sample_n(std::begin(left_indices), std::end(left_indices), left_count, m_gen);
    }
    if (right_count > 0)
    {
        sample_n(std::begin(right_indices), std::end(right_indices), right_count, m_gen);
    }

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
