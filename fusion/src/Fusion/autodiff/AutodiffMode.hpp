#ifndef AUTODIFF_MODE_HPP
#define AUTODIFF_MODE_HPP

#include <memory>

#include "EngineContext.hpp"

template <typename T> class ADTensor;

namespace autodiff {
// NOLINTBEGIN(cppcoreguidelines-avoid-non-const-global-variables)
inline thread_local bool g_enable_grad = true;
// NOLINTEND(cppcoreguidelines-avoid-non-const-global-variables)

class NoGradGuard {
 public:
   NoGradGuard() : prev_(g_enable_grad) { g_enable_grad = false; }

   NoGradGuard(const NoGradGuard &) = delete;
   NoGradGuard &operator=(const NoGradGuard &) = delete;

   NoGradGuard(NoGradGuard &&) = delete;
   NoGradGuard &operator=(NoGradGuard &&) = delete;

   bool prev() const { return prev_; }

   ~NoGradGuard() { g_enable_grad = prev(); }

 private:
   bool prev_;
};

inline bool grad_enabled() { return g_enable_grad; }

template <typename T> inline bool should_trace(const ADTensor<T> &x) {
   return grad_enabled() && x.requires_grad() && EngineContext<T>::has();
}

template <typename T>
inline bool should_trace(const ADTensor<T> &x, const ADTensor<T> &y) {
   return grad_enabled() && (x.requires_grad() || y.requires_grad()) &&
          EngineContext<T>::has();
}
} // namespace autodiff

#endif // AUTODIFF_MODE_HPP
