#include <condition_variable>
#include <memory>
#include <mutex>
#include <queue>

template <typename T> class ThreadSafeQueue {
private:
  mutable std::mutex mutex_;
  std::queue<T> queue_;
  std::condition_variable condition_;

public:
  ThreadSafeQueue() {}
  void push(const T &value) {
    std::lock_guard<std::mutex> lock(mutex_);
    queue_.push(value);
    condition_.notify_one();
  }
  void
  wait_and_pop(T &value) { // does this need to be a refernece?? why not return
    std::unique_lock<std::mutex> lock(mutex_);
    condition_.wait(lock, [this] { return !queue_.empty(); });
    value = std::move(queue_.front());
    queue_.pop();
  }

  std::shared_ptr<T> wait_and_pop() {
    std::unique_lock<std::mutex> lock(mutex_);
    condition_.wait(lock, [this] { return !queue_.empty(); });
    std::shared_ptr<T> res = std::make_shared<T>(std::move(queue_.front()));
    queue_.pop();
    return res;
  }

  std::shared_ptr<T> try_pop() {
    std::unique_lock<std::mutex> lock(mutex_);
    if (queue_.empty()) {
      return std::shared_ptr<T>();
    };
    std::shared_ptr<T> res = std::make_shared<T>(std::move(queue_.front()));
    queue_.pop();
    return res;
  }
};
