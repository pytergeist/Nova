#include "xsimd/xsimd.hpp"
#include <cmath>
#include <cstddef>

namespace unary_ops {
struct exp {
  template <class C, class Tag, class Arch>
  void operator()(Arch, const C &a, C &res, Tag) {
    using b_type = xsimd::batch<double, Arch>;
    std::size_t inc = b_type::size;
    std::size_t size = res.size();
    std::size_t vec_size = size - size % inc;
    for (std::size_t i = 0; i < vec_size; i += inc) {
      b_type avec = b_type::load(&a[i], Tag());
      b_type rvec = xsimd::exp(avec);
      xsimd::store(&res[i], rvec, Tag());
    }
    for (std::size_t i = vec_size; i < size; ++i) {
      res[i] = std::exp(a[i]);
    }
  }
};

struct sqrt {
  template <class C, class Tag, class Arch>
  void operator()(Arch, const C &a, C &res, Tag) {
    using b_type = xsimd::batch<double, Arch>;
    std::size_t inc = b_type::size;
    std::size_t size = res.size();
    std::size_t vec_size = size - size % inc;
    for (std::size_t i = 0; i < vec_size; i += inc) {
      b_type avec = b_type::load(&a[i], Tag());
      b_type rvec = xsimd::sqrt(avec);
      xsimd::store(&res[i], rvec, Tag());
    }
    for (std::size_t i = vec_size; i < size; ++i) {
      res[i] = std::sqrt(a[i]);
    }
  }
};

struct log {
  template <class C, class Tag, class Arch>
  void operator()(Arch, const C &a, C &res, Tag) {
    using b_type = xsimd::batch<double, Arch>;
    std::size_t inc = b_type::size;
    std::size_t size = res.size();
    std::size_t vec_size = size - size % inc;
    for (std::size_t i = 0; i < vec_size; i += inc) {
      b_type avec = b_type::load(&a[i], Tag());
      b_type rvec = xsimd::log(avec);
      xsimd::store(&res[i], rvec, Tag());
    }
    for (std::size_t i = vec_size; i < size; ++i) {
      res[i] = std::log(a[i]);
    }
  }
};

} // namespace unary_ops
