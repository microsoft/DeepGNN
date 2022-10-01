// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "threadpool.h"

namespace snark
{

ThreadPool::ThreadPool() : m_stop(false)
{
    Initialize(0);
}

ThreadPool::ThreadPool(std::size_t thread_count) : m_stop(false)
{
    Initialize(thread_count);
}

void ThreadPool::Initialize(std::size_t thread_count)
{
    if (thread_count == 0)
    {
        thread_count = std::thread::hardware_concurrency();
    }

    auto worker_func = [this]() {
        while (!m_stop)
        {
            std::function<void()> task;
            {
                std::unique_lock<std::mutex> lock(m_mutex);
                m_condition.wait(lock, [this] { return m_stop || !m_tasks.empty(); });
                if (m_stop)
                {
                    return;
                }

                task = std::move(m_tasks.front());
                m_tasks.pop();
            }

            if (task)
            {
                task();
            }
        }
    };

    for (std::size_t i = 0; i < thread_count; ++i)
    {
        m_workers.push_back(std::thread(worker_func));
    }
}

std::shared_ptr<std::promise<void>> ThreadPool::Submit(std::function<void()> callback)
{
    auto wait_handler = std::make_shared<std::promise<void>>();
    m_tasks.emplace([callback, wait_handler]() {
        callback();
        wait_handler->set_value();
    });

    m_condition.notify_one();
    return wait_handler;
}

ThreadPool::~ThreadPool()
{
    m_stop = true;
    m_condition.notify_all();

    for (auto &worker : m_workers)
    {
        if (worker.joinable())
        {
            worker.join();
        }
    }
}

} // namespace snark
