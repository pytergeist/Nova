#pragma once
#include "xsimd/xsimd.hpp"
#include <cstddef>

namespace reduction_ops {
struct sum {
  template <class Arch, class C>
  void operator()(Arch, const C &in, C &out) const {
    using T = typename C::value_type;
    using batch_t = xsimd::batch<T, Arch>;

    batch_t acc(T(0));

    std::size_t size = in.size();
    std::size_t n = size / batch_t::size * batch_t::size;
    const T *ptr = in.data();

    for (std::size_t i = 0; i < n; i += batch_t::size) {
      acc += batch_t::load_unaligned(ptr + i);
    }

    T result = xsimd::hadd(acc);

    for (std::size_t i = n; i < size; ++i) {
      result += ptr[i];
    }

    out[0] = result;
  }
};
} // namespace reduction_ops
