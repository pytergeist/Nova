#ifndef FUSION_VEC128_NEON_HPP
#define FUSION_VEC128_NEON_HPP

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <sleef.h> // NOLINT

#if defined(FUSION_ENABLE_NEON) &&                                             \
    (defined(__ARM_NEON) || defined(__ARM_NEON__))
#include <arm_neon.h>
#endif

#include "Fusion/common/Hints.hpp"
#include "vec/Vec128NeonBackend.hpp"
#include "vec/VecLoop.hpp"

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
// Core contiguous kernels
// =========================
// All assume: a, b, dst are contiguous T buffers of length n.

template <typename T> // TODO: fix this impl
inline void sum_f32_neon(T *__restrict dst, const T *__restrict a,
                         std::size_t n) {
#if defined(FUSION_ENABLE_NEON) &&                                             \
    (defined(__ARM_NEON) || defined(__ARM_NEON__))
   std::size_t i = 0;
   float32x4_t acc0 = vdupq_n_f32(0.0f);
   float32x4_t acc1 = vdupq_n_f32(0.0f);
   float32x4_t acc2 = vdupq_n_f32(0.0f);
   float32x4_t acc3 = vdupq_n_f32(0.0f);

   for (; i + kBlock <= n; i += kBlock) {
      float32x4_t v0 = vld1q_f32(a + i + 0 * kF32Lanes);
      float32x4_t v1 = vld1q_f32(a + i + 1 * kF32Lanes);
      float32x4_t v2 = vld1q_f32(a + i + 2 * kF32Lanes);
      float32x4_t v3 = vld1q_f32(a + i + 3 * kF32Lanes);
      acc0 = vaddq_f32(acc0, v0);
      acc1 = vaddq_f32(acc1, v1);
      acc2 = vaddq_f32(acc2, v2);
      acc3 = vaddq_f32(acc3, v3);
   }

   float32x4_t acc = vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3));

   for (; i + kStep <= n; i += kStep) {
      float32x4_t v = vld1q_f32(a + i);
      acc = vaddq_f32(acc, v);
   }

   T sum;
#if defined(__aarch64__)
   sum = vaddvq_f32(acc); // horizontal add (AArch64)
#else
   float32x2_t s2 = vadd_f32(vget_low_f32(acc), vget_high_f32(acc));
   s2 = vpadd_f32(s2, s2);
   sum = vget_lane_f32(s2, 0);
#endif

   for (; i < n; ++i)
      sum += a[i];

   *dst = sum;
#else
   T sum = 0.0f;
   for (std::size_t i = 0; i < n; ++i)
      sum += a[i];
   *dst = sum;
#endif
}

template <typename T>
inline void sqrt_f32_neon(T *__restrict dst, const T *__restrict a,
                          std::size_t n) {

   using B = Neon128<T>;
   return simd::detail::unary_contiguous_apply<T, B>(
       dst, a, n, [](B::vec vx) -> B::vec { return B::sqrt(vx); },
       [](T x) -> T { return std::sqrt(x); });
}

template <typename T>
inline void exp_f32_neon(T *__restrict dst, const T *__restrict a,
                         std::size_t n) {

   using B = Neon128<T>;
   return simd::detail::unary_contiguous_apply<T, B>(
       dst, a, n, [](B::vec vx) -> B::vec { return B::exp(vx); },
       [](T x) -> T { return std::exp(x); });
}

template <typename T>
inline void log_f32_neon(T *__restrict dst, const T *__restrict a,
                         std::size_t n) {

   using B = Neon128<T>;
   return simd::detail::unary_contiguous_apply<T, B>(
       dst, a, n, [](B::vec vx) -> B::vec { return B::log(vx); },
       [](T x) -> T { return std::log(x); });
}

template <typename T>
inline void pow_f32_neon(T *__restrict dst, const T *__restrict a,
                         const T *__restrict b, std::size_t n) {

   using B = Neon128<T>;
   return simd::detail::binary_contiguous_apply<T, B>(
       dst, a, b, n,
       [](B::vec vx, B::vec vy) -> B::vec { return B::pow(vx, vy); },
       [](T x, T y) -> T { return std::pow(x, y); });
}

template <typename T>
inline void maximum_f32_neon(T *__restrict dst, const T *__restrict a,
                             const T *__restrict b, std::size_t n) {

   using B = Neon128<T>;
   return simd::detail::binary_contiguous_apply<T, B>(
       dst, a, b, n,
       [](B::vec vx, B::vec vy) -> B::vec { return B::maximum(vx, vy); },
       [](T x, T y) -> T { return x > y ? x : y; });
}

template <typename T>
inline void greater_than_f32_neon(T *__restrict dst, const T *__restrict a,
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
inline void greater_than_equal_f32_neon(T *__restrict dst,
                                        const T *__restrict a,
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
inline void add_f32_neon(T *__restrict dst, const T *__restrict a,
                         const T *__restrict b, std::size_t n) {

   using B = Neon128<T>;
   return simd::detail::binary_contiguous_apply<T, B>(
       dst, a, b, n,
       [](B::vec vx, B::vec vy) -> B::vec { return B::add(vx, vy); },
       [](T x, T y) -> T { return x + y; });
}

template <typename T>
inline void sub_f32_neon(T *__restrict dst, const T *__restrict a,
                         const T *__restrict b, std::size_t n) {

   using B = Neon128<T>;
   return simd::detail::binary_contiguous_apply<T, B>(
       dst, a, b, n,
       [](B::vec vx, B::vec vy) -> B::vec { return B::sub(vx, vy); },
       [](T x, T y) -> T { return x - y; });
}

template <typename T>
inline void mul_f32_neon(T *__restrict dst, const T *__restrict a,
                         const T *__restrict b, std::size_t n) {

   using B = Neon128<T>;
   return simd::detail::binary_contiguous_apply<T, B>(
       dst, a, b, n,
       [](B::vec vx, B::vec vy) -> B::vec { return B::mul(vx, vy); },
       [](T x, T y) -> T { return x * y; });
}

template <typename T>
inline void div_f32_neon(T *__restrict dst, const T *__restrict a,
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
inline void greater_than_f32_neon_scalar(T *__restrict dst,
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
inline void greater_than_equal_f32_neon_scalar(T *__restrict dst,
                                               const T *__restrict a, const T b,
                                               std::size_t n) {

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
inline void pow_f32_neon_scalar(T *__restrict dst, const T *__restrict a,
                                const T b, std::size_t n) {

   using B = Neon128<T>;
   return simd::detail::binary_contiguous_scalar_apply<T, B>(
       dst, a, b, n,
       [](B::vec vx, B::vec vy) -> B::vec { return B::pow(vx, vy); },
       [](T x, T y) -> T { return std::pow(x, y); });
}

template <typename T>
inline void maximum_f32_neon_scalar(T *__restrict dst, const T *__restrict a,
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
inline void add_f32_neon_scalar(T *__restrict dst, const T *__restrict a,
                                const T b, std::size_t n) {

   using B = Neon128<T>;
   return simd::detail::binary_contiguous_scalar_apply<T, B>(
       dst, a, b, n,
       [](B::vec vx, B::vec vy) -> B::vec { return B::add(vx, vy); },
       [](T x, T y) -> T { return x + y; });
}

template <typename T>
inline void sub_f32_neon_scalar(T *__restrict dst, const T *__restrict a,
                                const T b, std::size_t n) {

   using B = Neon128<T>;
   return simd::detail::binary_contiguous_scalar_apply<T, B>(
       dst, a, b, n,
       [](B::vec vx, B::vec vy) -> B::vec { return B::sub(vx, vy); },
       [](T x, T y) -> T { return x - y; });
}

template <typename T>
inline void mul_f32_neon_scalar(T *__restrict dst, const T *__restrict a,
                                const T b, std::size_t n) {

   using B = Neon128<T>;
   return simd::detail::binary_contiguous_scalar_apply<T, B>(
       dst, a, b, n,
       [](B::vec vx, B::vec vy) -> B::vec { return B::mul(vx, vy); },
       [](T x, T y) -> T { return x * y; });
}

template <typename T>
inline void div_f32_neon_scalar(T *__restrict dst, const T *__restrict a,
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

#endif // FUSION_VEC128_NEON_HPP
