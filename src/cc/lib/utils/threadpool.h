// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#ifndef THREAD_POOL_H
#define THREAD_POOL_H

#include <condition_variable>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <thread>
#include <vector>

namespace snark
{

class ThreadPool
{
  public:
    ThreadPool();
    ThreadPool(size_t);

    template <class F, class... Args>
    auto Submit(F &&f, Args &&...args) -> std::future<typename std::result_of<F(Args...)>::type>;

    ~ThreadPool();

  private:
    void Initialize(size_t threads);

    std::vector<std::thread> m_workers;
    std::queue<std::function<void()>> m_tasks;
    std::mutex m_queue_mutex;
    std::condition_variable m_condition;
    bool m_stop;
};

inline ThreadPool::ThreadPool() : m_stop(false)
{
    Initialize(0);
}

inline ThreadPool::ThreadPool(size_t threads) : m_stop(false)
{
    Initialize(threads);
}

inline void ThreadPool::Initialize(size_t threads)
{
    if (threads == 0)
    {
        threads = std::thread::hardware_concurrency();
    }

    for (size_t i = 0; i < threads; ++i)
    {
        m_workers.emplace_back([this] {
            for (;;)
            {
                std::function<void()> task;

                {
                    std::unique_lock<std::mutex> lock(m_queue_mutex);
                    m_condition.wait(lock, [this] { return m_stop || !m_tasks.empty(); });
                    if (m_stop && m_tasks.empty())
                    {
                        return;
                    }

                    task = std::move(m_tasks.front());
                    m_tasks.pop();
                }

                task();
            }
        });
    }
}

template <class F, class... Args>
auto ThreadPool::Submit(F &&f, Args &&...args) -> std::future<typename std::result_of<F(Args...)>::type>
{
    using return_type = typename std::result_of<F(Args...)>::type;

    auto task =
        std::make_shared<std::packaged_task<return_type()>>(std::bind(std::forward<F>(f), std::forward<Args>(args)...));

    std::future<return_type> res = task->get_future();
    {
        std::unique_lock<std::mutex> lock(m_queue_mutex);
        if (m_stop)
        {
            throw std::runtime_error("Submit on stopped ThreadPool");
        }

        m_tasks.emplace([task]() { (*task)(); });
    }

    m_condition.notify_one();
    return res;
}

inline ThreadPool::~ThreadPool()
{
    {
        std::unique_lock<std::mutex> lock(m_queue_mutex);
        m_stop = true;
    }

    m_condition.notify_all();

    for (auto &worker : m_workers)
    {
        worker.join();
    }
}

} // namespace snark

#endif
