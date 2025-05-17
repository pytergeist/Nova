#include "xsimd/xsimd.hpp"
#include <cmath>
#include <cstddef>

namespace xsimd_ops {
// Binary Operations
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

struct pow {
  template <class C, class Tag, class Arch>
  void operator()(Arch, const C &a, const C &b, C &res, Tag) {
    using b_type = xsimd::batch<double, Arch>;
    std::size_t inc = b_type::size;
    std::size_t size = res.size();
    std::size_t vec_size = size - size % inc;
    for (std::size_t i = 0; i < vec_size; i += inc) {
      b_type avec = b_type::load(&a[i], Tag());
      b_type bvec = b_type::load(&b[i], Tag());
      b_type rvec = xsimd::pow(avec, bvec);
      xsimd::store(&res[i], rvec, Tag());
    }
    for (std::size_t i = vec_size; i < size; ++i) {
      res[i] = std::pow(a[i], b[i]);
    }
  }
};

// Unary Operations

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

// Reduction ops
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
} // namespace xsimd_ops
