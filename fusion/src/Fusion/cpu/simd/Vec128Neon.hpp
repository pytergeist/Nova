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
static constexpr std::size_t kNeonVectorBytes = 16;
static constexpr std::size_t kF32Lanes = kNeonVectorBytes / sizeof(float);
static constexpr std::size_t kUnroll = 4;
static constexpr std::size_t kBlock = kUnroll * kF32Lanes; // 16

// Constants for scalar tail
static constexpr std::size_t kStepVec = kBlock;
static constexpr std::size_t kStep = kUnroll;

#if defined(FUSION_ENABLE_NEON) &&                                             \
    (defined(__ARM_NEON) || defined(__ARM_NEON__))
// =========================
// Core contiguous kernels
// =========================
// All assume: a, b, dst are contiguous float buffers of length n.

inline void sum_f32_neon(float *__restrict dst, const float *__restrict a,
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

   float sum;
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
   float sum = 0.0f;
   for (std::size_t i = 0; i < n; ++i)
      sum += a[i];
   *dst = sum;
#endif
}

inline void sqrt_f32_neon(float *__restrict dst, const float *__restrict a,
                          std::size_t n) {

   using B = Neon128<float>;
   return simd::detail::unary_contiguous_apply<float, B>(
       dst, a, n, [](B::vec vx) -> B::vec { return B::sqrt(vx); },
       [](float x) -> float { return std::sqrt(x); });
}

inline void exp_f32_neon(float *__restrict dst, const float *__restrict a,
                         std::size_t n) {

   using B = Neon128<float>;
   return simd::detail::unary_contiguous_apply<float, B>(
       dst, a, n, [](B::vec vx) -> B::vec { return B::exp(vx); },
       [](float x) -> float { return std::exp(x); });
}

inline void log_f32_neon(float *__restrict dst, const float *__restrict a,
                         std::size_t n) {

   using B = Neon128<float>;
   return simd::detail::unary_contiguous_apply<float, B>(
       dst, a, n, [](B::vec vx) -> B::vec { return B::log(vx); },
       [](float x) -> float { return std::log(x); });
}

inline void pow_f32_neon(float *__restrict dst, const float *__restrict a,
                         const float *__restrict b, std::size_t n) {

   using B = Neon128<float>;
   return simd::detail::binary_contiguous_apply<float, B>(
       dst, a, b, n,
       [](B::vec vx, B::vec vy) -> B::vec { return B::pow(vx, vy); },
       [](float x, float y) -> float { return std::pow(x, y); });
}

inline void maximum_f32_neon(float *__restrict dst, const float *__restrict a,
                             const float *__restrict b, std::size_t n) {

   using B = Neon128<float>;
   return simd::detail::binary_contiguous_apply<float, B>(
       dst, a, b, n,
       [](B::vec vx, B::vec vy) -> B::vec { return B::maximum(vx, vy); },
       [](float x, float y) -> float { return x > y ? x : y; });
}

inline void greater_than_f32_neon(float *__restrict dst,
                                  const float *__restrict a,
                                  const float *__restrict b, std::size_t n) {

   using B = Neon128<float>;
   return simd::detail::binary_contiguous_apply<float, B>(
       dst, a, b, n,
       [](B::vec vx, B::vec vy) -> B::vec {
          return B::blend(B::cgt(vx, vy), B::duplicate(1.0f),
                          B::duplicate(0.0f));
       },
       [](float x, float y) -> float { return x > y; });
}

inline void greater_than_equal_f32_neon(float *__restrict dst,
                                        const float *__restrict a,
                                        const float *__restrict b,
                                        std::size_t n) {

   using B = Neon128<float>;
   return simd::detail::binary_contiguous_apply<float, B>(
       dst, a, b, n,
       [](B::vec vx, B::vec vy) -> B::vec {
          return B::blend(B::cge(vx, vy), B::duplicate(1.0f),
                          B::duplicate(0.0f));
       },
       [](float x, float y) -> float { return x >= y; });
}

inline void add_f32_neon(float *__restrict dst, const float *__restrict a,
                         const float *__restrict b, std::size_t n) {

   using B = Neon128<float>;
   return simd::detail::binary_contiguous_apply<float, B>(
       dst, a, b, n,
       [](B::vec vx, B::vec vy) -> B::vec { return B::add(vx, vy); },
       [](float x, float y) -> float { return x + y; });
}

inline void sub_f32_neon(float *__restrict dst, const float *__restrict a,
                         const float *__restrict b, std::size_t n) {

   using B = Neon128<float>;
   return simd::detail::binary_contiguous_apply<float, B>(
       dst, a, b, n,
       [](B::vec vx, B::vec vy) -> B::vec { return B::sub(vx, vy); },
       [](float x, float y) -> float { return x - y; });
}

inline void mul_f32_neon(float *__restrict dst, const float *__restrict a,
                         const float *__restrict b, std::size_t n) {

   using B = Neon128<float>;
   return simd::detail::binary_contiguous_apply<float, B>(
       dst, a, b, n,
       [](B::vec vx, B::vec vy) -> B::vec { return B::mul(vx, vy); },
       [](float x, float y) -> float { return x * y; });
}

inline void div_f32_neon(float *__restrict dst, const float *__restrict a,
                         const float *__restrict b, std::size_t n) {

   using B = Neon128<float>;
   return simd::detail::binary_contiguous_apply<float, B>(
       dst, a, b, n,
       [](B::vec vx, B::vec vy) -> B::vec { return B::div(vx, vy); },
       [](float x, float y) -> float { return x / y; });
}

// =========================
// Scalar-RHS wrappers
// ========================= // TODO: Rename from RHS
// Useful when your inner dim sees stride==0 for RHS/LHS (broadcast scalar).
// If LHS is the scalar instead, you can either add "scalar_lhs" variants
// or just swap operands in the caller for commutative ops.

inline void greater_than_f32_neon_scalar(float *__restrict dst,
                                         const float *__restrict a,
                                         const float b, std::size_t n) {

   using B = Neon128<float>;
   return simd::detail::binary_contiguous_scalar_apply<float, B>(
       dst, a, b, n,
       [](B::vec vx, B::vec vy) -> B::vec {
          return B::blend(B::cgt(vx, vy), B::duplicate(1.0f),
                          B::duplicate(0.0f));
       },
       [](float x, float y) -> float { return x > y; });
}

inline void greater_than_equal_f32_neon_scalar(float *__restrict dst,
                                               const float *__restrict a,
                                               const float b, std::size_t n) {

   using B = Neon128<float>;
   return simd::detail::binary_contiguous_scalar_apply<float, B>(
       dst, a, b, n,
       [](B::vec vx, B::vec vy) -> B::vec {
          return B::blend(B::cge(vx, vy), B::duplicate(1.0f),
                          B::duplicate(0.0f));
       },
       [](float x, float y) -> float { return x >= y; });
}

inline void pow_f32_neon_scalar(float *__restrict dst,
                                const float *__restrict a, const float b,
                                std::size_t n) {

   using B = Neon128<float>;
   return simd::detail::binary_contiguous_scalar_apply<float, B>(
       dst, a, b, n,
       [](B::vec vx, B::vec vy) -> B::vec { return B::pow(vx, vy); },
       [](float x, float y) -> float { return std::pow(x, y); });
}

inline void maximum_f32_neon_scalar(float *__restrict dst,
                                    const float *__restrict a, const float b,
                                    std::size_t n) {

   using B = Neon128<float>;
   return simd::detail::binary_contiguous_scalar_apply<float, B>(
       dst, a, b, n,
       [](B::vec vx, B::vec vy) -> B::vec { return B::maximum(vx, vy); },
       [](float x, float y) -> float {
          return x > y ? x : y;
          ;
       });
}

inline void add_f32_neon_scalar(float *__restrict dst,
                                const float *__restrict a, const float b,
                                std::size_t n) {

   using B = Neon128<float>;
   return simd::detail::binary_contiguous_scalar_apply<float, B>(
       dst, a, b, n,
       [](B::vec vx, B::vec vy) -> B::vec { return B::add(vx, vy); },
       [](float x, float y) -> float { return x + y; });
}

inline void sub_f32_neon_scalar(float *__restrict dst,
                                const float *__restrict a, const float b,
                                std::size_t n) {

   using B = Neon128<float>;
   return simd::detail::binary_contiguous_scalar_apply<float, B>(
       dst, a, b, n,
       [](B::vec vx, B::vec vy) -> B::vec { return B::sub(vx, vy); },
       [](float x, float y) -> float { return x - y; });
}

inline void mul_f32_neon_scalar(float *__restrict dst,
                                const float *__restrict a, const float b,
                                std::size_t n) {

   using B = Neon128<float>;
   return simd::detail::binary_contiguous_scalar_apply<float, B>(
       dst, a, b, n,
       [](B::vec vx, B::vec vy) -> B::vec { return B::mul(vx, vy); },
       [](float x, float y) -> float { return x * y; });
}

inline void div_f32_neon_scalar(float *__restrict dst,
                                const float *__restrict a, const float b,
                                std::size_t n) {

   using B = Neon128<float>;
   return simd::detail::binary_contiguous_scalar_apply<float, B>(
       dst, a, b, n,
       [](B::vec vx, B::vec vy) -> B::vec { return B::div(vx, vy); },
       [](float x, float y) -> float { return x / y; });
}
#else // --------- Fallback (non-NEON builds) ---------

#include "VecFallback.hpp

#endif
} // namespace simd

#endif // FUSION_VEC128_NEON_HPP
