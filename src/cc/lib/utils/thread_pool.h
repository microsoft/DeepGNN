// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#ifndef SNARK_THREAD_POOL_H
#define SNARK_THREAD_POOL_H

#include "boost/asio.hpp"
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

    template <typename _Iterator, typename _Function>
    void RunInParallel(_Iterator _First, _Iterator _Last, const _Function &_Func)
    {
        typedef typename ::std::iterator_traits<_Iterator>::difference_type _Index_type;

        _Index_type _Range_size = _Last - _First;
        _Index_type _I;

        std::size_t index = 0;
        std::vector<std::shared_ptr<std::promise<void>>> promise_list;
        // Split the available work in chunks
        for (_I = 0; _I < _Range_size; _I++)
        {
            auto p = std::make_shared<std::promise<void>>();
            boost::asio::post(m_thread_pool, [=]() {
                _Func(index, _First[_I]);
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

  private:
    ThreadPool(const std::size_t &size);

    static std::size_t m_thread_size;
    boost::asio::thread_pool m_thread_pool;
};

} // namespace snark

#endif // SNARK_THREAD_POOL_H
