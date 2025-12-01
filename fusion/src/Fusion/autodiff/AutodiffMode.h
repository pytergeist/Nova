#ifndef AUTODIFF_MODE_H
#define AUTODIFF_MODE_H

#include <memory>

#include "EngineContext.h"

template <typename T> class ADTensor;

namespace autodiff {
inline thread_local bool g_enable_grad = true;

struct NoGradGuard {
   bool prev_;
   NoGradGuard() : prev_(g_enable_grad) { g_enable_grad = false; }
   ~NoGradGuard() { g_enable_grad = prev_; }
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

#endif // AUTODIFF_MODE_H
