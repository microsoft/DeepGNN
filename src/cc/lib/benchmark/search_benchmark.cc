// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <algorithm>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include <benchmark/benchmark.h>

#ifdef SNARK_PLATFORM_LINUX
#include <mimalloc-override.h>
#endif

static void BM_LOWER_BOUND(benchmark::State &state)
{
    size_t max_count = state.range(0);
    std::vector<int64_t> elements(max_count);
    std::iota(elements.begin(), elements.end(), 0);
    std::vector<int64_t> ids(elements);
    int64_t seed = 42;
    std::shuffle(ids.begin(), ids.end(), std::mt19937_64(seed));
    size_t curr = 0;
    for (auto _ : state)
    {
        ++curr;
        if (curr == max_count)
        {
            curr = 0;
        }
        const auto res = std::lower_bound(elements.cbegin(), elements.cend(), ids[curr]);
        if (*res != ids[curr])
        {
            throw std::runtime_error("not found" + std::to_string(ids[curr]) + " got " + std::to_string(*res));
        }
    }
}

static void BM_STD_FIND(benchmark::State &state)
{
    size_t max_count = state.range(0);
    std::vector<int64_t> elements(max_count);
    std::iota(elements.begin(), elements.end(), 0);
    std::vector<int64_t> ids(elements);
    int64_t seed = 42;
    std::shuffle(ids.begin(), ids.end(), std::mt19937_64(seed));
    size_t curr = 0;
    for (auto _ : state)
    {
        ++curr;
        if (curr == max_count)
        {
            curr = 0;
        }
        const auto res = std::find(elements.cbegin(), elements.cend(), ids[curr]);
        if (*res != ids[curr])
        {
            throw std::runtime_error("not found" + std::to_string(ids[curr]) + " got " + std::to_string(*res));
        }
    }
}

BENCHMARK(BM_LOWER_BOUND)->RangeMultiplier(4)->Range(1 << 3, 1 << 10);
BENCHMARK(BM_STD_FIND)->RangeMultiplier(4)->Range(1 << 3, 1 << 10);

BENCHMARK_MAIN();
