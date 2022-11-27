// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#ifndef SNARK_THREAD_POOL_H
#define SNARK_THREAD_POOL_H

#include "boost/asio.hpp"
#include <functional>
#include <future>
#include <memory>

namespace snark
{

class ThreadPool
{
  public:
    ThreadPool(const ThreadPool &) = delete;
    ThreadPool &operator=(const ThreadPool &) = delete;
    ThreadPool(ThreadPool &&) = delete;
    ThreadPool &operator=(ThreadPool &&) = delete;

    // Before calling GetInstance, we can use this to set the desired count of threads
    // in the thread pool, by default, hardware concurrency is used.
    static void SetThreadPoolSize(const std::size_t &size);
    static ThreadPool &GetInstance();

    template <typename _Range, typename _Function> void RunInParallel(_Range _RangeItem, const _Function &_Func)
    {
        std::size_t index = 0;
        std::vector<std::shared_ptr<std::promise<void>>> promise_list;
        // Split the available work in chunks
        for (auto &_Item : _RangeItem)
        {
            auto p = std::make_shared<std::promise<void>>();
            boost::asio::post(m_thread_pool, [=]() {
                _Func(index, _Item);
                p->set_value();
            });
            promise_list.push_back(p);
            index++;
        }

        for (auto p : promise_list)
        {
            p->get_future().get();
        }
    }

    void Submit(const std::function<void()> &callback);

  private:
    ThreadPool(const std::size_t &size);

    static std::size_t m_thread_size;
    boost::asio::thread_pool m_thread_pool;
};

} // namespace snark

#endif // SNARK_THREAD_POOL_H
