#include "xsimd/xsimd.hpp"
#include <cmath>
#include <cstddef>

namespace xsimd_ops {
    std::pair<bool, bool> is_scalar(size_t na, size_t nb) {
        return {na == 1, nb == 1};
    }

    template<class T, class Arch, class Tag, class VecOp, class ScalarOp>
    void simd_broadcast_binary(const T *a, size_t na, const T *b, size_t nb, T *r,
                               size_t nr, Arch arch, Tag tag, VecOp vec_op,
                               ScalarOp scalar_op) {
        using batch_t = xsimd::batch<T, Arch>;
        constexpr std::size_t N = batch_t::size;

        bool a_is_scalar = (na == 1), b_is_scalar = (nb == 1);
        std::size_t vec_end = nr - (nr % N);

        batch_t a0{}, b0{};
        if (a_is_scalar)
            a0 = batch_t::broadcast(a[0]);
        if (b_is_scalar)
            b0 = batch_t::broadcast(b[0]);

        for (std::size_t i = 0; i < vec_end; i += N) {
            batch_t va = a_is_scalar ? a0 : batch_t::load(a + i, tag);
            batch_t vb = b_is_scalar ? b0 : batch_t::load(b + i, tag);
            batch_t vr = vec_op(va, vb);
            xsimd::store(r + i, vr, tag);
        }

        for (std::size_t i = vec_end; i < nr; ++i) {
            T aval = a_is_scalar ? a[0] : a[i];
            T bval = b_is_scalar ? b[0] : b[i];
            r[i] = scalar_op(aval, bval);
        }
    }

    struct add {
        template<class T, class Arch, class Tag>
        void operator()(Arch arch, const T *a, size_t na, const T *b, size_t nb, T *r,
                        size_t nr, Tag tag) const {
            simd_broadcast_binary<T, Arch, Tag>(
                a, na, b, nb, r, nr, arch, tag, [](auto x, auto y) { return x + y; },
                [](T x, T y) { return x + y; });
        }
    };

    struct divide {
        template<class T, class Arch, class Tag>
        void operator()(Arch arch, const T *a, size_t na, const T *b, size_t nb, T *r,
                        size_t nr, Tag tag) const {
            simd_broadcast_binary<T, Arch, Tag>(
                a, na, b, nb, r, nr, arch, tag, [](auto x, auto y) { return x / y; },
                [](T x, T y) { return x / y; });
        }
    };

    struct subtract {
        template<class T, class Arch, class Tag>
        void operator()(Arch arch, const T *a, size_t na, const T *b, size_t nb, T *r,
                        size_t nr, Tag tag) const {
            simd_broadcast_binary<T, Arch, Tag>(
                a, na, b, nb, r, nr, arch, tag, [](auto x, auto y) { return x - y; },
                [](T x, T y) { return x - y; });
        }
    };

    struct multiply {
        template<class T, class Arch, class Tag>
        void operator()(Arch arch, const T *a, size_t na, const T *b, size_t nb, T *r,
                        size_t nr, Tag tag) const {
            simd_broadcast_binary<T, Arch, Tag>(
                a, na, b, nb, r, nr, arch, tag, [](auto x, auto y) { return x * y; },
                [](T x, T y) { return x * y; });
        }
    };

    struct pow {
        template<class T, class Arch, class Tag>
        void operator()(Arch arch, const T *a, size_t na, const T *b, size_t nb, T *r,
                        size_t nr, Tag tag) const {
            simd_broadcast_binary<T, Arch, Tag>(
                a, na, b, nb, r, nr, arch, tag,
                [](auto x, auto y) { return xsimd::pow(x, y); },
                [](T x, T y) { return std::pow(x, y); });
        }
    };


    struct greater_than_equal_to_bool {
        template<class T, class Arch, class Tag>
        void operator()(Arch arch,
                        const T *a, size_t na,
                        const T *b, size_t nb,
                        T *r, size_t nr,
                        Tag tag) const {
            using batch_t = xsimd::batch<T, Arch>;
            simd_broadcast_binary<T, Arch, Tag>(
                a, na, b, nb, r, nr, arch, tag,

                [](batch_t x, batch_t y) {
                    return batch_t(x >= y);
                },

                [](T x, T y) {
                    return T(x >= y);
                }
            );
        }
    };

    struct maximum {
        template<class T, class Arch, class Tag>
        void operator()(Arch arch, const T *a, size_t na, const T *b, size_t nb, T *r,
                        size_t nr, Tag tag) const {
            simd_broadcast_binary<T, Arch, Tag>(
                a, na, b, nb, r, nr, arch, tag,
                [](auto x, auto y) { return xsimd::max(x, y); },
                [](T x, T y) { return std::max(x, y); });
        }
    };

    // Unary Operations

    struct exp {
        template<class C, class Tag, class Arch>
        void operator()(Arch, const C &a, C &res, Tag) {
            using b_type = xsimd::batch<float, Arch>;
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
        template<class C, class Tag, class Arch>
        void operator()(Arch, const C &a, C &res, Tag) {
            using b_type = xsimd::batch<float, Arch>;
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
        template<class C, class Tag, class Arch>
        void operator()(Arch, const C &a, C &res, Tag) {
            using b_type = xsimd::batch<float, Arch>;
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
        template<class Arch, class C>
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

            T result = xsimd::reduce_add(acc);

            for (std::size_t i = n; i < size; ++i) {
                result += ptr[i];
            }

            out[0] = result;
        }
    };
} // namespace xsimd_ops
