#ifndef FUSION_CPU_VEC_NEON128_HPP
#define FUSION_CPU_VEC_NEON128_HPP

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <sleef.h> // NOLINT

#if defined(FUSION_ENABLE_NEON) &&                                             \
    (defined(__ARM_NEON) || defined(__ARM_NEON__))
#include <arm_neon.h>
#endif

#include "Fusion/common/Hints.hpp"
#include "backend/BackendNeon128.hpp"
#include "backend/VecLoop.hpp"

namespace simd {
// TODO: remove once sum fixed
static constexpr std::size_t kNeonVectorBytes = 16;
static constexpr std::size_t kF32Lanes = kNeonVectorBytes / sizeof(float);
static constexpr std::size_t kUnroll = 4;
static constexpr std::size_t kBlock = kUnroll * kF32Lanes; // 16

// Constants for scalar tail
static constexpr std::size_t kStepVec = kBlock;
static constexpr std::size_t kStep = kUnroll;

// TODO: we DO NOT support lhs side scalar operations - MAKE SURE you deal with
// non-commutative OPS!

#if defined(FUSION_ENABLE_NEON) &&                                             \
    (defined(__ARM_NEON) || defined(__ARM_NEON__))
// =========================
// Core contiguous kernels - Current alignment in fixed 64 // TODO: Fix alignment criteria?
// =========================
// All assume: a, b, dst are contiguous T buffers of length n.

template <typename T>
inline void sum_contiguous(T *__restrict dst, const T *__restrict a,
                           std::size_t n) {

   using B = Neon128<T>;
   return simd::detail::reduce_contiguous_apply<T, B>(
       dst, a, n, [](B::vec vx, B::vec vy) -> B::vec { return B::add(vx, vy); },
       [](B::vec vx) -> T { return B::horizontal_add(vx); },
       [](T acc, T x) -> T { return acc + x; });
}

template <typename T>
inline void sqrt_contiguous(T *__restrict dst, const T *__restrict a,
                            std::size_t n) {

   using B = Neon128<T>;
   return simd::detail::unary_contiguous_apply<T, B>(
       dst, a, n, [](B::vec vx) -> B::vec { return B::sqrt(vx); },
       [](T x) -> T { return std::sqrt(x); });
}

template <typename T>
inline void exp_contiguous(T *__restrict dst, const T *__restrict a,
                           std::size_t n) {

   using B = Neon128<T>;
   return simd::detail::unary_contiguous_apply<T, B>(
       dst, a, n, [](B::vec vx) -> B::vec { return B::exp(vx); },
       [](T x) -> T { return std::exp(x); });
}

template <typename T>
inline void log_contiguous(T *__restrict dst, const T *__restrict a,
                           std::size_t n) {

   using B = Neon128<T>;
   return simd::detail::unary_contiguous_apply<T, B>(
       dst, a, n, [](B::vec vx) -> B::vec { return B::log(vx); },
       [](T x) -> T { return std::log(x); });
}

template <typename T>
inline void pow_contiguous(T *__restrict dst, const T *__restrict a,
                           const T *__restrict b, std::size_t n) {

   using B = Neon128<T>;
   return simd::detail::binary_contiguous_apply<T, B>(
       dst, a, b, n,
       [](B::vec vx, B::vec vy) -> B::vec { return B::pow(vx, vy); },
       [](T x, T y) -> T { return std::pow(x, y); });
}

template <typename T>
inline void maximum_contiguous(T *__restrict dst, const T *__restrict a,
                               const T *__restrict b, std::size_t n) {

   using B = Neon128<T>;
   return simd::detail::binary_contiguous_apply<T, B>(
       dst, a, b, n,
       [](B::vec vx, B::vec vy) -> B::vec { return B::maximum(vx, vy); },
       [](T x, T y) -> T { return x > y ? x : y; });
}

template <typename T>
inline void greater_than_contiguous(T *__restrict dst, const T *__restrict a,
                                    const T *__restrict b, std::size_t n) {

   using B = Neon128<T>;
   return simd::detail::binary_contiguous_apply<T, B>(
       dst, a, b, n,
       [](B::vec vx, B::vec vy) -> B::vec {
          return B::blend(B::cgt(vx, vy), B::duplicate(1.0f),
                          B::duplicate(0.0f));
       },
       [](T x, T y) -> T { return x > y; });
}

template <typename T>
inline void
greater_than_equal_contiguous(T *__restrict dst, const T *__restrict a,
                              const T *__restrict b, std::size_t n) {

   using B = Neon128<T>;
   return simd::detail::binary_contiguous_apply<T, B>(
       dst, a, b, n,
       [](B::vec vx, B::vec vy) -> B::vec {
          return B::blend(B::cge(vx, vy), B::duplicate(1.0f),
                          B::duplicate(0.0f));
       },
       [](T x, T y) -> T { return x >= y; });
}

template <typename T>
inline void add_contiguous(T *__restrict dst, const T *__restrict a,
                           const T *__restrict b, std::size_t n) {

   using B = Neon128<T>;
   return simd::detail::binary_contiguous_apply<T, B>(
       dst, a, b, n,
       [](B::vec vx, B::vec vy) -> B::vec { return B::add(vx, vy); },
       [](T x, T y) -> T { return x + y; });
}

template <typename T>
inline void sub_contiguous(T *__restrict dst, const T *__restrict a,
                           const T *__restrict b, std::size_t n) {

   using B = Neon128<T>;
   return simd::detail::binary_contiguous_apply<T, B>(
       dst, a, b, n,
       [](B::vec vx, B::vec vy) -> B::vec { return B::sub(vx, vy); },
       [](T x, T y) -> T { return x - y; });
}

template <typename T>
inline void mul_contiguous(T *__restrict dst, const T *__restrict a,
                           const T *__restrict b, std::size_t n) {

   using B = Neon128<T>;
   return simd::detail::binary_contiguous_apply<T, B>(
       dst, a, b, n,
       [](B::vec vx, B::vec vy) -> B::vec { return B::mul(vx, vy); },
       [](T x, T y) -> T { return x * y; });
}

template <typename T>
inline void div_contiguous(T *__restrict dst, const T *__restrict a,
                           const T *__restrict b, std::size_t n) {

   using B = Neon128<T>;
   return simd::detail::binary_contiguous_apply<T, B>(
       dst, a, b, n,
       [](B::vec vx, B::vec vy) -> B::vec { return B::div(vx, vy); },
       [](T x, T y) -> T { return x / y; });
}

// =========================
// Scalar wrappers
// =========================
// Useful when your inner dim sees stride==0 for RHS/LHS (broadcast scalar).
// If LHS is the scalar instead, you can either add "scalar_lhs" variants
// or just swap operands in the caller for commutative ops.

template <typename T>
inline void greater_than_contiguous_scalar(T *__restrict dst,
                                           const T *__restrict a, const T b,
                                           std::size_t n) {

   using B = Neon128<T>;
   return simd::detail::binary_contiguous_scalar_apply<T, B>(
       dst, a, b, n,
       [](B::vec vx, B::vec vy) -> B::vec {
          return B::blend(B::cgt(vx, vy), B::duplicate(1.0f),
                          B::duplicate(0.0f));
       },
       [](T x, T y) -> T { return x > y; });
}

template <typename T>
inline void greater_than_equal_contiguous_scalar(T *__restrict dst,
                                                 const T *__restrict a,
                                                 const T b, std::size_t n) {

   using B = Neon128<T>;
   return simd::detail::binary_contiguous_scalar_apply<T, B>(
       dst, a, b, n,
       [](B::vec vx, B::vec vy) -> B::vec {
          return B::blend(B::cge(vx, vy), B::duplicate(1.0f),
                          B::duplicate(0.0f));
       },
       [](T x, T y) -> T { return x >= y; });
}

template <typename T>
inline void pow_contiguous_scalar(T *__restrict dst, const T *__restrict a,
                                  const T b, std::size_t n) {

   using B = Neon128<T>;
   return simd::detail::binary_contiguous_scalar_apply<T, B>(
       dst, a, b, n,
       [](B::vec vx, B::vec vy) -> B::vec { return B::pow(vx, vy); },
       [](T x, T y) -> T { return std::pow(x, y); });
}

template <typename T>
inline void maximum_contiguous_scalar(T *__restrict dst, const T *__restrict a,
                                      const T b, std::size_t n) {

   using B = Neon128<T>;
   return simd::detail::binary_contiguous_scalar_apply<T, B>(
       dst, a, b, n,
       [](B::vec vx, B::vec vy) -> B::vec { return B::maximum(vx, vy); },
       [](T x, T y) -> T {
          return x > y ? x : y;
          ;
       });
}

template <typename T>
inline void add_contiguous_scalar(T *__restrict dst, const T *__restrict a,
                                  const T b, std::size_t n) {

   using B = Neon128<T>;
   return simd::detail::binary_contiguous_scalar_apply<T, B>(
       dst, a, b, n,
       [](B::vec vx, B::vec vy) -> B::vec { return B::add(vx, vy); },
       [](T x, T y) -> T { return x + y; });
}

template <typename T>
inline void sub_contiguous_scalar(T *__restrict dst, const T *__restrict a,
                                  const T b, std::size_t n) {

   using B = Neon128<T>;
   return simd::detail::binary_contiguous_scalar_apply<T, B>(
       dst, a, b, n,
       [](B::vec vx, B::vec vy) -> B::vec { return B::sub(vx, vy); },
       [](T x, T y) -> T { return x - y; });
}

template <typename T>
inline void mul_contiguous_scalar(T *__restrict dst, const T *__restrict a,
                                  const T b, std::size_t n) {

   using B = Neon128<T>;
   return simd::detail::binary_contiguous_scalar_apply<T, B>(
       dst, a, b, n,
       [](B::vec vx, B::vec vy) -> B::vec { return B::mul(vx, vy); },
       [](T x, T y) -> T { return x * y; });
}

template <typename T>
inline void div_contiguous_scalar(T *__restrict dst, const T *__restrict a,
                                  const T b, std::size_t n) {

   using B = Neon128<T>;
   return simd::detail::binary_contiguous_scalar_apply<T, B>(
       dst, a, b, n,
       [](B::vec vx, B::vec vy) -> B::vec { return B::div(vx, vy); },
       [](T x, T y) -> T { return x / y; });
}
#else // --------- Fallback (non-NEON builds) ---------

#include "VecFallback.hpp"

#endif
} // namespace simd

#endif // FUSION_CPU_VEC_NEON128_HPP
