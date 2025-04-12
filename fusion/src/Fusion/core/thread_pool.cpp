#include <atomic>
#include <thread>
#include <vector>

#include "thread_safe_queue.h"

class JoinThreads {
  std::vector<std::thread> &threads;

public:
  explicit JoinThreads(std::vector<std::thread> &threads_)
      : threads(threads_) {}
  ~JoinThreads() {
    for (auto &t : threads) {
      t.join();
    }
  }
};

class ThreadPool {
  std::atomic_bool done_;
  ThreadSafeQueue<std::function<void()>> queue_;
  std::vector<std::thread> threads_;
  JoinThreads join_threads_;
  void worker_thread() {
    while (!done_) {
      std::function<void()> task;
      if (queue_.try_pop(task)) {
        task();
      } else {
        std::this_thread::yield();
      }
    }
  }

public:
  ThreadPool() : done_(false), join_threads_(threads_) {
    auto const thread_count = std::thread::hardware_concurrency();
    try {
      for (auto i = 0; i < thread_count; ++i) {
        threads_.push_back(std::thread(&ThreadPool::worker_thread, this));
      }
    } catch (...) {
      done_ = true;
      throw;
    }
  }
  ~ThreadPool() { done_ = true; }
  template <typename FunctionType> void submit(FunctionType function) {
    queue_.push(std::function<void()>(function));
  }
};
