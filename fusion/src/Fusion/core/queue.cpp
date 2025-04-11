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
  void wait_and_pop(T &value) {
    std::unique_lock<std::mutex> lock(mutex);
    condition.wait(lock, [this] { return !queue.empty(); });
    value = std::move(queue.front());
    queue.pop();
  }
};
