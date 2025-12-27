#ifndef THREAD_POOL_H
#define THREAD_POOL_H

#include <atomic>
#include <functional>
#include <thread>
#include <vector>

#include "ThreadSafeQueue.hpp"

// RAII helper: joins all threads on destruction
class JoinThreads {
 public:
   explicit JoinThreads(std::vector<std::thread> &threads);
   ~JoinThreads();

   JoinThreads(const JoinThreads &) = delete;
   JoinThreads &operator=(const JoinThreads &) = delete;

 private:
   std::vector<std::thread> &threads_;
};

class ThreadPool {
 public:
   ThreadPool();
   ~ThreadPool();

   ThreadPool(const ThreadPool &) = delete;
   ThreadPool &operator=(const ThreadPool &) = delete;

   // Non-templated public API
   void submit(std::function<void()> task);

 private:
   void worker_thread();

   std::atomic_bool done_{false};
   ThreadSafeQueue<std::function<void()>> queue_;
   std::vector<std::thread> threads_;
   JoinThreads join_threads_;
};

#endif // THREAD_POOL_H
