// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#ifndef SNARK_THREAD_POOL_H
#define SNARK_THREAD_POOL_H

#include <condition_variable>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

namespace snark
{

class ThreadPool
{
  public:
    // default constructor will create threads using hardware concurrency.
    ThreadPool();

    // user can specify the thread count.
    ThreadPool(std::size_t thread_count);

    // stop all worker thread.
    ~ThreadPool();

    // submit a job to the thread pool, user can use promise to wait the job
    // to be finished.
    std::shared_ptr<std::promise<void>> Submit(std::function<void()> callback);

  private:
    // create threads in the pool.
    void Initialize(std::size_t thread_count);

    std::vector<std::thread> m_workers;
    std::queue<std::function<void()>> m_tasks;
    std::mutex m_mutex;
    std::condition_variable m_condition;
    bool m_stop;
};

} // namespace snark

#endif
