#include <cassert>
#include <cmath>

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

} // namespace snark
