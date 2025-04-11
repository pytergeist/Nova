#include <condition_variable>
#include <mutex>
#include <queue>

template <typename T> class ThreadSafeQueue {
private:
  mutable std::mutex mutex;
  std::queue<T> queue;
  std::condition_variable condition;

public:
  ThreadSafeQueue() {}
  void push(const T &value) {
    std::lock_guard<std::mutex> lock(mutex);
    queue.push(value);
    condition.notify_one();
  }
};
