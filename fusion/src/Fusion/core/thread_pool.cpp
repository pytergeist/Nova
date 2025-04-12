#include <thread>
#include <vector>

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
