// Code copied from: https://prng.di.unimi.it/xoroshiro128plusplus.c
// and wrapped as a bit generator for a random module.

/*  Written in 2019 by David Blackman and Sebastiano Vigna (vigna@acm.org)
To the extent possible under law, the author has dedicated all copyright
and related and neighboring rights to this software to the public domain
worldwide. This software is distributed without any warranty.
See <http://creativecommons.org/publicdomain/zero/1.0/>. */
#ifndef XOROSHIRO_H
#define XOROSHIRO_H

#include <cassert>
#include <cstdint>
#include <functional>
#include <limits>

namespace snark
{
namespace detail
{

class SplitMix64Generator
{
  public:
    explicit SplitMix64Generator(uint64_t seed = 0)
    {
        this->seed(seed);
    }

    SplitMix64Generator(const SplitMix64Generator &other) = default;

    inline uint64_t operator()()
    {
        return Transform(m_state);
    }

    inline void seed(uint64_t seed)
    {
        m_state = seed;
    }

  private:
    static inline uint64_t Transform(uint64_t &state)
    {
        uint64_t z = (state += 0x9e3779b97f4a7c15);
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9;
        z = (z ^ (z >> 27)) * 0x94d049bb133111eb;
        return z ^ (z >> 31);
    }
    uint64_t m_state = 0;
};
struct Xoroshiro128PlusGeneratorState
{
    uint64_t data[2];
};

static_assert(sizeof(Xoroshiro128PlusGeneratorState) == sizeof(uint64_t[2]), "");

} // namespace detail

class Xoroshiro128PlusGenerator
{
  public:
    Xoroshiro128PlusGenerator(const Xoroshiro128PlusGenerator &other) = default;
    explicit Xoroshiro128PlusGenerator(uint64_t seed)
    {
        this->seed(seed);
    }

    using result_type = uint64_t;
    inline result_type operator()()
    {
        return Transform(m_state);
    }

    void seed(uint64_t seed)
    {
        Convert(seed, m_state);
    }

    static constexpr result_type min()
    {
        return std::numeric_limits<uint64_t>::min();
    }

    static constexpr result_type max()
    {
        return std::numeric_limits<uint64_t>::max();
    }

    bool operator==(const Xoroshiro128PlusGenerator &other) const
    {
        bool result = std::equal(reinterpret_cast<const std::uint8_t *>(&m_state),
                                 reinterpret_cast<const std::uint8_t *>(&m_state) + sizeof(m_state),
                                 reinterpret_cast<const std::uint8_t *>(&other.m_state),
                                 reinterpret_cast<const std::uint8_t *>(&other.m_state) + sizeof(other.m_state));
        return result;
    }

    bool operator!=(const Xoroshiro128PlusGenerator &other) const
    {
        return !(*this == other);
    }

  private:
    inline void Convert(uint64_t seed, detail::Xoroshiro128PlusGeneratorState &state)
    {
        detail::SplitMix64Generator stateGen{seed};
        state.data[0] = stateGen();
        state.data[1] = stateGen();
    }

    inline std::uint64_t _rotl64(uint64_t x, const int k)
    {
        return (x << k) | (x >> (64 - k));
    }

    inline uint64_t Transform(detail::Xoroshiro128PlusGeneratorState &state)
    {
        uint64_t s0 = state.data[0];
        std::uint64_t s1 = state.data[1];
        uint64_t result = s0 + s1;

        s1 ^= s0;
        state.data[0] = _rotl64(s0, 24) ^ s1 ^ (s1 << 16); // a, b
        state.data[1] = _rotl64(s1, 37);                   // c

        return result;
    }

    detail::Xoroshiro128PlusGeneratorState m_state = {};
};
} // namespace snark

#endif
