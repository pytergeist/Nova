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

struct maximum {
  template <class C, class Tag, class Arch>
  void operator()(Arch, const C &a, const C &b, C &res, Tag) {
    using b_type = xsimd::batch<double, Arch>;
    std::size_t inc = b_type::size;
    std::size_t size = res.size();
    std::size_t vec_size = size - size % inc;
    for (std::size_t i = 0; i < vec_size; i += inc) {
      b_type avec = b_type::load(&a[i], Tag());
      b_type bvec = b_type::load(&b[i], Tag());
      b_type rvec = xsimd::max(avec, bvec);
      xsimd::store(&res[i], rvec, Tag());
    }
    for (std::size_t i = vec_size; i < size; ++i) {
      res[i] = std::max(a[i], b[i]);
    }
  }
};
} // namespace reduction_ops
