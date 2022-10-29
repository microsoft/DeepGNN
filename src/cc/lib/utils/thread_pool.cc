// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "thread_pool.h"

namespace snark
{

std::size_t ThreadPool::m_thread_size = 0;
void ThreadPool::SetThreadPoolSize(const std::size_t &size)
{
    m_thread_size = size;
}

ThreadPool &ThreadPool::GetInstance()
{
    if (m_thread_size == 0)
    {
        m_thread_size = std::thread::hardware_concurrency();
    }

    static ThreadPool instance(m_thread_size);
    return instance;
}

ThreadPool::ThreadPool(const std::size_t &size) : m_thread_pool(size)
{
}

} // namespace snark
