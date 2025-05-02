#include "xsimd/xsimd.hpp"
#include <cstddef>

namespace binary_ops {
struct add {
  template <class C, class Tag, class Arch>
  void operator()(Arch, const C &a, const C &b, C &res, Tag) {
    using b_type = xsimd::batch<double, Arch>;
    std::size_t inc = b_type::size;
    std::size_t size = res.size();
    // size for which the vectorization is possible
    std::size_t vec_size = size - size % inc;
    for (std::size_t i = 0; i < vec_size; i += inc) {
      b_type avec = b_type::load(&a[i], Tag());
      b_type bvec = b_type::load(&b[i], Tag());
      b_type rvec = avec + bvec;
      xsimd::store(&res[i], rvec, Tag());
    }
    // Remaining part that cannot be vectorize
    for (std::size_t i = vec_size; i < size; ++i) {
      res[i] = a[i] + b[i];
    }
  }
};

struct divide {
  template <class C, class Tag, class Arch>
  void operator()(Arch, const C &a, const C &b, C &res, Tag) {
    using b_type = xsimd::batch<double, Arch>;
    std::size_t inc = b_type::size;
    std::size_t size = res.size();
    std::size_t vec_size = size - size % inc;
    for (std::size_t i = 0; i < vec_size; i += inc) {
      b_type avec = b_type::load(&a[i], Tag());
      b_type bvec = b_type::load(&b[i], Tag());
      b_type rvec = avec / bvec;
      xsimd::store(&res[i], rvec, Tag());
    }
    for (std::size_t i = vec_size; i < size; ++i) {
      res[i] = a[i] / b[i];
    }
  }
};

struct subtract {
  template <class C, class Tag, class Arch>
  void operator()(Arch, const C &a, const C &b, C &res, Tag) {
    using b_type = xsimd::batch<double, Arch>;
    std::size_t inc = b_type::size;
    std::size_t size = res.size();
    std::size_t vec_size = size - size % inc;
    for (std::size_t i = 0; i < vec_size; i += inc) {
      b_type avec = b_type::load(&a[i], Tag());
      b_type bvec = b_type::load(&b[i], Tag());
      b_type rvec = avec - bvec;
      xsimd::store(&res[i], rvec, Tag());
    }
    for (std::size_t i = vec_size; i < size; ++i) {
      res[i] = a[i] - b[i];
    }
  }
};

struct multiply {
  template <class C, class Tag, class Arch>
  void operator()(Arch, const C &a, const C &b, C &res, Tag) {
    using b_type = xsimd::batch<double, Arch>;
    std::size_t inc = b_type::size;
    std::size_t size = res.size();
    std::size_t vec_size = size - size % inc;
    for (std::size_t i = 0; i < vec_size; i += inc) {
      b_type avec = b_type::load(&a[i], Tag());
      b_type bvec = b_type::load(&b[i], Tag());
      b_type rvec = avec * bvec;
      xsimd::store(&res[i], rvec, Tag());
    }
    for (std::size_t i = vec_size; i < size; ++i) {
      res[i] = a[i] * b[i];
    }
  }
};
} // namespace binary_ops
