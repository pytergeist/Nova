#pragma once
#ifndef FUSION_VEC128_NEON_H
#define FUSION_VEC128_NEON_H

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <sleef.h> // NOLINT

#if defined(FUSION_ENABLE_NEON) &&                                             \
    (defined(__ARM_NEON) || defined(__ARM_NEON__))
#include <arm_neon.h>
#endif

#include "Fusion/common/Hints.h"

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

inline void sum_f32_neon(float* __restrict dst, const float* __restrict a, std::size_t n) {
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

inline void sqrt_f32_neon(float* __restrict dst, const float* __restrict a, std::size_t n) {
   std::size_t i = 0;

   const float * __restrict pa = a;
   float * __restrict pd = dst;

   FUSION_CONST_ASSUME_ALIGNED(float, pa, 64);
   FUSION_ASSUME_ALIGNED(float, pd, 64);

   for (; i + kBlock <= n; i += kBlock) {
      float32x4x4_t va = vld1q_f32_x4(pa); pa += kBlock;

      va.val[0] = Sleef_sqrtf4_u05(va.val[0]);
      va.val[1] = Sleef_sqrtf4_u05(va.val[1]);
      va.val[2] = Sleef_sqrtf4_u05(va.val[2]);
      va.val[3] = Sleef_sqrtf4_u05(va.val[3]);

      vst1q_f32_x4(pd, va); pd += kBlock;

   }
   for (; i + kStep <= n; i += kStep) {
      float32x4_t va = vld1q_f32(pa); pa += kStep;
      vst1q_f32(pd, Sleef_sqrtf4_u05(va)); pd += kStep;
   }
   for (; i < n; ++i)
      *pd++ = std::sqrt(*pa++);
}


inline void exp_f32_neon(float* __restrict dst, const float* __restrict a, std::size_t n) {
   std::size_t i = 0;

   const float * __restrict pa = a;
   float * __restrict pd = dst;

   FUSION_CONST_ASSUME_ALIGNED(float, pa, 64);
   FUSION_ASSUME_ALIGNED(float, pd, 64);

   for (; i + kBlock <= n; i += kBlock) {
      float32x4x4_t va = vld1q_f32_x4(pa); pa += kBlock;

      va.val[0] = Sleef_expf4_u10(va.val[0]);
      va.val[1] = Sleef_expf4_u10(va.val[1]);
      va.val[2] = Sleef_expf4_u10(va.val[2]);
      va.val[3] = Sleef_expf4_u10(va.val[3]);

      vst1q_f32_x4(pd, va); pd += kBlock;

   }
   for (; i + kStep <= n; i += kStep) {
      float32x4_t va = vld1q_f32(pa); pa += kStep;
      vst1q_f32(pd, Sleef_expf4_u10(va)); pd += kStep;
   }
   for (; i < n; ++i)
      *pd++ = std::exp(*pa++);
}

inline void log_f32_neon(float* __restrict dst, const float* __restrict a, std::size_t n) {
   std::size_t i = 0;

   const float * __restrict pa = a;
   float * __restrict pd = dst;

   FUSION_CONST_ASSUME_ALIGNED(float, pa, 64);
   FUSION_ASSUME_ALIGNED(float, pd, 64);

   for (; i + kBlock <= n; i += kBlock) {
      float32x4x4_t va = vld1q_f32_x4(pa); pa += kBlock;

      va.val[0] = Sleef_logf4_u10(va.val[0]);
      va.val[1] = Sleef_logf4_u10(va.val[1]);
      va.val[2] = Sleef_logf4_u10(va.val[2]);
      va.val[3] = Sleef_logf4_u10(va.val[3]);

      vst1q_f32_x4(pd, va); pd += kBlock;

   }
   for (; i + kStep <= n; i += kStep) {
      float32x4_t va = vld1q_f32(pa); pa += kStep;
      vst1q_f32(pd, Sleef_logf4_u10(va)); pd += kStep;
   }
   for (; i < n; ++i)
      *pd++ = std::log(*pa++);
}

inline void pow_f32_neon(float* __restrict dst, const float* __restrict a, const float* __restrict b,
                         std::size_t n) {
   std::size_t i = 0;

   const float * __restrict pa = a;
   const float * __restrict pb = b;
   float * __restrict pd = dst;

   FUSION_CONST_ASSUME_ALIGNED(float, pa, 64);
   FUSION_CONST_ASSUME_ALIGNED(float, pb, 64);
   FUSION_ASSUME_ALIGNED(float, pd, 64);

   for (; i + kBlock <= n; i += kBlock) {
      float32x4x4_t va = vld1q_f32_x4(pa); pa += kBlock;
   	  float32x4x4_t vb = vld1q_f32_x4(pb); pb += kBlock;

      va.val[0] = Sleef_powf4_u10(va.val[0], vb.val[0]);
      va.val[1] = Sleef_powf4_u10(va.val[1], vb.val[1]);
      va.val[2] = Sleef_powf4_u10(va.val[2], vb.val[2]);
      va.val[3] = Sleef_powf4_u10(va.val[3], vb.val[3]);

      vst1q_f32_x4(pd, va); pd += kBlock;

   }
   for (; i + kStep <= n; i += kStep) {
      float32x4_t va = vld1q_f32(pa); pa += kStep;
      float32x4_t vb = vld1q_f32(pb); pb += kStep;
      vst1q_f32(pd, Sleef_powf4_u10(va, vb)); pd += kStep;
   }
   for (; i < n; ++i)
      *pd++ = std::pow(*pa++, *pb++);
}

inline void maximum_f32_neon(float* __restrict dst, const float* __restrict a, const float* __restrict b,
                             std::size_t n) {
   std::size_t i = 0;

   for (; i + kBlock <= n; i += kBlock) {
      float32x4_t a0 = vld1q_f32(a + i + 0 * kF32Lanes);
      float32x4_t a1 = vld1q_f32(a + i + 1 * kF32Lanes);
      float32x4_t a2 = vld1q_f32(a + i + 2 * kF32Lanes);
      float32x4_t a3 = vld1q_f32(a + i + 3 * kF32Lanes);

      float32x4_t b0 = vld1q_f32(b + i + 0 * kF32Lanes);
      float32x4_t b1 = vld1q_f32(b + i + 1 * kF32Lanes);
      float32x4_t b2 = vld1q_f32(b + i + 2 * kF32Lanes);
      float32x4_t b3 = vld1q_f32(b + i + 3 * kF32Lanes);

      vst1q_f32(dst + i + 0 * kF32Lanes, vmaxq_f32(a0, b0));
      vst1q_f32(dst + i + 1 * kF32Lanes, vmaxq_f32(a1, b1));
      vst1q_f32(dst + i + 2 * kF32Lanes, vmaxq_f32(a2, b2));
      vst1q_f32(dst + i + 3 * kF32Lanes, vmaxq_f32(a3, b3));
   }
   for (; i + kStep <= n; i += kStep) {
      float32x4_t va = vld1q_f32(a + i);
      float32x4_t vb = vld1q_f32(b + i);
      vst1q_f32(dst + i, vmaxq_f32(va, vb));
   }
   for (; i < n; ++i)
      dst[i] = a[i] > b[i] ? a[i] : b[i];
}

inline void greater_than_f32_neon(float* __restrict dst, const float* __restrict a, const float* __restrict b,
                                  std::size_t n) {
   std::size_t i = 0;

   for (; i + kBlock <= n; i += kBlock) {
      float32x4_t a0 = vld1q_f32(a + i + 0 * kF32Lanes);
      float32x4_t a1 = vld1q_f32(a + i + 1 * kF32Lanes);
      float32x4_t a2 = vld1q_f32(a + i + 2 * kF32Lanes);
      float32x4_t a3 = vld1q_f32(a + i + 3 * kF32Lanes);

      float32x4_t b0 = vld1q_f32(b + i + 0 * kF32Lanes);
      float32x4_t b1 = vld1q_f32(b + i + 1 * kF32Lanes);
      float32x4_t b2 = vld1q_f32(b + i + 2 * kF32Lanes);
      float32x4_t b3 = vld1q_f32(b + i + 3 * kF32Lanes);

      uint32x4_t mask0 = vcgtq_f32(a0, b0);
      vst1q_f32(dst + i + 0 * kF32Lanes,
                vbslq_f32(mask0, vdupq_n_f32(1.0f), vdupq_n_f32(0.0f)));
      uint32x4_t mask1 = vcgtq_f32(a1, b1);
      vst1q_f32(dst + i + 1 * kF32Lanes,
                vbslq_f32(mask1, vdupq_n_f32(1.0f), vdupq_n_f32(0.0f)));
      uint32x4_t mask2 = vcgtq_f32(a2, b2);
      vst1q_f32(dst + i + 2 * kF32Lanes,
                vbslq_f32(mask2, vdupq_n_f32(1.0f), vdupq_n_f32(0.0f)));
      uint32x4_t mask3 = vcgtq_f32(a3, b3);
      vst1q_f32(dst + i + 3 * kF32Lanes,
                vbslq_f32(mask3, vdupq_n_f32(1.0f), vdupq_n_f32(0.0f)));
   }
   for (; i + kStep <= n; i += kStep) {
      float32x4_t va = vld1q_f32(a + i);
      float32x4_t vb = vld1q_f32(b + i);
      uint32x4_t mask_va = vcgtq_f32(va, vb);
      vst1q_f32(dst + i,
                vbslq_f32(mask_va, vdupq_n_f32(1.0f), vdupq_n_f32(0.0f)));
   }
   for (; i < n; ++i)
      dst[i] = a[i] > b[i];
}

inline void greater_than_equal_f32_neon(float* __restrict dst, const float* __restrict a,
                                        const float* __restrict b, std::size_t n) {
   std::size_t i = 0;

   for (; i + kBlock <= n; i += kBlock) {
      float32x4_t a0 = vld1q_f32(a + i + 0 * kF32Lanes);
      float32x4_t a1 = vld1q_f32(a + i + 1 * kF32Lanes);
      float32x4_t a2 = vld1q_f32(a + i + 2 * kF32Lanes);
      float32x4_t a3 = vld1q_f32(a + i + 3 * kF32Lanes);

      float32x4_t b0 = vld1q_f32(b + i + 0 * kF32Lanes);
      float32x4_t b1 = vld1q_f32(b + i + 1 * kF32Lanes);
      float32x4_t b2 = vld1q_f32(b + i + 2 * kF32Lanes);
      float32x4_t b3 = vld1q_f32(b + i + 3 * kF32Lanes);

      uint32x4_t mask0 = vcgeq_f32(a0, b0);
      vst1q_f32(dst + i + 0 * kF32Lanes,
                vbslq_f32(mask0, vdupq_n_f32(1.0f), vdupq_n_f32(0.0f)));
      uint32x4_t mask1 = vcgeq_f32(a1, b1);
      vst1q_f32(dst + i + 1 * kF32Lanes,
                vbslq_f32(mask1, vdupq_n_f32(1.0f), vdupq_n_f32(0.0f)));
      uint32x4_t mask2 = vcgeq_f32(a2, b2);
      vst1q_f32(dst + i + 2 * kF32Lanes,
                vbslq_f32(mask2, vdupq_n_f32(1.0f), vdupq_n_f32(0.0f)));
      uint32x4_t mask3 = vcgeq_f32(a3, b3);
      vst1q_f32(dst + i + 3 * kF32Lanes,
                vbslq_f32(mask3, vdupq_n_f32(1.0f), vdupq_n_f32(0.0f)));
   }
   for (; i + kStep <= n; i += kStep) {
      float32x4_t va = vld1q_f32(a + i);
      float32x4_t vb = vld1q_f32(b + i);
      uint32x4_t mask_va = vcgeq_f32(va, vb);
      vst1q_f32(dst + i,
                vbslq_f32(mask_va, vdupq_n_f32(1.0f), vdupq_n_f32(0.0f)));
   }
   for (; i < n; ++i)
      dst[i] = a[i] >= b[i];
}

inline void add_f32_neon(float* __restrict dst, const float* __restrict a, const float* __restrict b,
                         std::size_t n) {
   std::size_t i = 0;

   const float * __restrict pa = a;
   const float * __restrict pb = b;
   float * __restrict pd = dst;

   FUSION_CONST_ASSUME_ALIGNED(float, pa, 64);
   FUSION_CONST_ASSUME_ALIGNED(float, pb, 64);
   FUSION_ASSUME_ALIGNED(float, pd, 64);

   for (; i + kBlock <= n; i += kBlock) {
      float32x4x4_t va = vld1q_f32_x4(pa); pa += kBlock;
   	  float32x4x4_t vb = vld1q_f32_x4(pb); pb += kBlock;

      va.val[0] = vaddq_f32(va.val[0], vb.val[0]);
      va.val[1] = vaddq_f32(va.val[1], vb.val[1]);
      va.val[2] = vaddq_f32(va.val[2], vb.val[2]);
      va.val[3] = vaddq_f32(va.val[3], vb.val[3]);

      vst1q_f32_x4(pd, va); pd += kBlock;

   }
   for (; i + kStep <= n; i += kStep) {
      float32x4_t va = vld1q_f32(pa); pa += kStep;
      float32x4_t vb = vld1q_f32(pb); pb += kStep;
      vst1q_f32(pd, vaddq_f32(va, vb)); pd += kStep;
   }
   for (; i < n; ++i)
      *pd++ = *pa++ + *pb++;
}

inline void sub_f32_neon(float* __restrict dst, const float* __restrict a, const float* __restrict b,
                         std::size_t n) {
   std::size_t i = 0;

   const float * __restrict pa = a;
   const float * __restrict pb = b;
   float * __restrict pd = dst;

   FUSION_CONST_ASSUME_ALIGNED(float, pa, 64);
   FUSION_CONST_ASSUME_ALIGNED(float, pb, 64);
   FUSION_ASSUME_ALIGNED(float, pd, 64);

   for (; i + kBlock <= n; i += kBlock) {
      float32x4x4_t va = vld1q_f32_x4(pa); pa += kBlock;
   	  float32x4x4_t vb = vld1q_f32_x4(pb); pb += kBlock;

      va.val[0] = vsubq_f32(va.val[0], vb.val[0]);
      va.val[1] = vsubq_f32(va.val[1], vb.val[1]);
      va.val[2] = vsubq_f32(va.val[2], vb.val[2]);
      va.val[3] = vsubq_f32(va.val[3], vb.val[3]);

      vst1q_f32_x4(pd, va); pd += kBlock;

   }
   for (; i + kStep <= n; i += kStep) {
      float32x4_t va = vld1q_f32(pa); pa += kStep;
      float32x4_t vb = vld1q_f32(pb); pb += kStep;
      vst1q_f32(pd, vsubq_f32(va, vb)); pd += kStep;
   }
   for (; i < n; ++i)
      *pd++ = *pa++ - *pb++;
}

inline void mul_f32_neon(float* __restrict dst, const float* __restrict a, const float* __restrict b,
                         std::size_t n) {
   std::size_t i = 0;

   const float * __restrict pa = a;
   const float * __restrict pb = b;
   float * __restrict pd = dst;

   FUSION_CONST_ASSUME_ALIGNED(float, pa, 64);
   FUSION_CONST_ASSUME_ALIGNED(float, pb, 64);
   FUSION_ASSUME_ALIGNED(float, pd, 64);

   for (; i + kBlock <= n; i += kBlock) {
      float32x4x4_t va = vld1q_f32_x4(pa); pa += kBlock;
   	  float32x4x4_t vb = vld1q_f32_x4(pb); pb += kBlock;

      va.val[0] = vmulq_f32(va.val[0], vb.val[0]);
      va.val[1] = vmulq_f32(va.val[1], vb.val[1]);
      va.val[2] = vmulq_f32(va.val[2], vb.val[2]);
      va.val[3] = vmulq_f32(va.val[3], vb.val[3]);

      vst1q_f32_x4(pd, va); pd += kBlock;

   }
   for (; i + kStep <= n; i += kStep) {
      float32x4_t va = vld1q_f32(pa); pa += kStep;
      float32x4_t vb = vld1q_f32(pb); pb += kStep;
      vst1q_f32(pd, vmulq_f32(va, vb)); pd += kStep;
   }
   for (; i < n; ++i)
      *pd++ = *pa++ * *pb++;
}

inline void div_f32_neon(float* __restrict dst, const float* __restrict a, const float* __restrict b,
                         std::size_t n) {
   std::size_t i = 0;

   const float * __restrict pa = a;
   const float * __restrict pb = b;
   float * __restrict pd = dst;

   FUSION_CONST_ASSUME_ALIGNED(float, pa, 64);
   FUSION_CONST_ASSUME_ALIGNED(float, pb, 64);
   FUSION_ASSUME_ALIGNED(float, pd, 64);

   for (; i + kBlock <= n; i += kBlock) {
      float32x4x4_t va = vld1q_f32_x4(pa); pa += kBlock;
   	  float32x4x4_t vb = vld1q_f32_x4(pb); pb += kBlock;

      va.val[0] = vdivq_f32(va.val[0], vb.val[0]);
      va.val[1] = vdivq_f32(va.val[1], vb.val[1]);
      va.val[2] = vdivq_f32(va.val[2], vb.val[2]);
      va.val[3] = vdivq_f32(va.val[3], vb.val[3]);

      vst1q_f32_x4(pd, va); pd += kBlock;

   }
   for (; i + kStep <= n; i += kStep) {
      float32x4_t va = vld1q_f32(pa); pa += kStep;
      float32x4_t vb = vld1q_f32(pb); pb += kStep;
      vst1q_f32(pd, vdivq_f32(va, vb)); pd += kStep;
   }
   for (; i < n; ++i)
      *pd++ = *pa++ / *pb++;
}

// =========================
// Scalar-RHS wrappers
// ========================= // TODO: Rename from RHS
// Useful when your inner dim sees stride==0 for RHS/LHS (broadcast scalar).
// If LHS is the scalar instead, you can either add "scalar_lhs" variants
// or just swap operands in the caller for commutative ops.

inline void greater_than_f32_neon_scalar(float* __restrict dst, const float* __restrict a,
                                         const float b, std::size_t n) {
   std::size_t i = 0;

   for (; i + kBlock <= n; i += kBlock) {
      float32x4_t a0 = vld1q_f32(a + i + 0 * kF32Lanes);
      float32x4_t a1 = vld1q_f32(a + i + 1 * kF32Lanes);
      float32x4_t a2 = vld1q_f32(a + i + 2 * kF32Lanes);
      float32x4_t a3 = vld1q_f32(a + i + 3 * kF32Lanes);

      float32x4_t vb = vdupq_n_f32(b);
      uint32x4_t mask0 = vcgtq_f32(a0, vb);
      vst1q_f32(dst + i + 0 * kF32Lanes,
                vbslq_f32(mask0, vdupq_n_f32(1.0f), vdupq_n_f32(0.0f)));
      uint32x4_t mask1 = vcgtq_f32(a1, vb);
      vst1q_f32(dst + i + 1 * kF32Lanes,
                vbslq_f32(mask1, vdupq_n_f32(1.0f), vdupq_n_f32(0.0f)));
      uint32x4_t mask2 = vcgtq_f32(a2, vb);
      vst1q_f32(dst + i + 2 * kF32Lanes,
                vbslq_f32(mask2, vdupq_n_f32(1.0f), vdupq_n_f32(0.0f)));
      uint32x4_t mask3 = vcgtq_f32(a3, vb);
      vst1q_f32(dst + i + 3 * kF32Lanes,
                vbslq_f32(mask3, vdupq_n_f32(1.0f), vdupq_n_f32(0.0f)));
   }
   for (; i + kStep <= n; i += kStep) {
      float32x4_t va = vld1q_f32(a + i);
      float32x4_t vb = vdupq_n_f32(b);
      uint32x4_t mask_va = vcgtq_f32(va, vb);
      vst1q_f32(dst + i,
                vbslq_f32(mask_va, vdupq_n_f32(1.0f), vdupq_n_f32(0.0f)));
   }
   for (; i < n; ++i)
      dst[i] = a[i] > b;
}

inline void greater_than_equal_f32_neon_scalar(float* __restrict dst, const float* __restrict a,
                                               const float b, std::size_t n) {
   std::size_t i = 0;

   for (; i + kBlock <= n; i += kBlock) {
      float32x4_t a0 = vld1q_f32(a + i + 0 * kF32Lanes);
      float32x4_t a1 = vld1q_f32(a + i + 1 * kF32Lanes);
      float32x4_t a2 = vld1q_f32(a + i + 2 * kF32Lanes);
      float32x4_t a3 = vld1q_f32(a + i + 3 * kF32Lanes);

      float32x4_t vb = vdupq_n_f32(b);
      uint32x4_t mask0 = vcgeq_f32(a0, vb);
      vst1q_f32(dst + i + 0 * kF32Lanes,
                vbslq_f32(mask0, vdupq_n_f32(1.0f), vdupq_n_f32(0.0f)));
      uint32x4_t mask1 = vcgeq_f32(a1, vb);
      vst1q_f32(dst + i + 1 * kF32Lanes,
                vbslq_f32(mask1, vdupq_n_f32(1.0f), vdupq_n_f32(0.0f)));
      uint32x4_t mask2 = vcgeq_f32(a2, vb);
      vst1q_f32(dst + i + 2 * kF32Lanes,
                vbslq_f32(mask2, vdupq_n_f32(1.0f), vdupq_n_f32(0.0f)));
      uint32x4_t mask3 = vcgeq_f32(a3, vb);
      vst1q_f32(dst + i + 3 * kF32Lanes,
                vbslq_f32(mask3, vdupq_n_f32(1.0f), vdupq_n_f32(0.0f)));
   }
   for (; i + kStep <= n; i += kStep) {
      float32x4_t va = vld1q_f32(a + i);
      float32x4_t vb = vdupq_n_f32(b);
      uint32x4_t mask_va = vcgeq_f32(va, vb);
      vst1q_f32(dst + i,
                vbslq_f32(mask_va, vdupq_n_f32(1.0f), vdupq_n_f32(0.0f)));
   }
   for (; i < n; ++i)
      dst[i] = a[i] >= b;
}

inline void pow_f32_neon_scalar_rhs(float* __restrict dst, const float* __restrict a, const float b,
                                    std::size_t n) {
   std::size_t i = 0;

   const float * __restrict pa = a;
   float32x4_t vb = vdupq_n_f32(b);
   float * __restrict pd = dst;

   FUSION_CONST_ASSUME_ALIGNED(float, pa, 64);
   FUSION_ASSUME_ALIGNED(float, pd, 64);

   for (; i + kBlock <= n; i += kBlock) {
      float32x4x4_t va = vld1q_f32_x4(pa); pa += kBlock;

      va.val[0] = Sleef_powf4_u10(va.val[0], vb);
      va.val[1] = Sleef_powf4_u10(va.val[1], vb);
      va.val[2] = Sleef_powf4_u10(va.val[2], vb);
      va.val[3] = Sleef_powf4_u10(va.val[3], vb);

      vst1q_f32_x4(pd, va); pd += kBlock;

   }
   for (; i + kStep <= n; i += kStep) {
      float32x4_t va = vld1q_f32(pa); pa += kStep;
      vst1q_f32(pd, Sleef_powf4_u10(va, vb)); pd += kStep;
   }
   for (; i < n; ++i)
      *pd++ = std::pow(*pa++, b);
  }

inline void maximum_f32_neon_scalar_rhs(float* __restrict dst, const float* __restrict a, float b,
                                        std::size_t n) {
   float32x4_t vb = vdupq_n_f32(b);
   std::size_t i = 0;
   for (; i + kBlock <= n; i += kBlock) {
      float32x4_t a0 = vld1q_f32(a + i + 0 * kF32Lanes);
      float32x4_t a1 = vld1q_f32(a + i + 1 * kF32Lanes);
      float32x4_t a2 = vld1q_f32(a + i + 2 * kF32Lanes);
      float32x4_t a3 = vld1q_f32(a + i + 3 * kF32Lanes);
      vst1q_f32(dst + i + 0 * kF32Lanes, vmaxq_f32(a0, vb));
      vst1q_f32(dst + i + 1 * kF32Lanes, vmaxq_f32(a1, vb));
      vst1q_f32(dst + i + 2 * kF32Lanes, vmaxq_f32(a2, vb));
      vst1q_f32(dst + i + 3 * kF32Lanes, vmaxq_f32(a3, vb));
   }
   for (; i + kStep <= n; i += kStep) {
      float32x4_t va = vld1q_f32(a + i);
      vst1q_f32(dst + i, vmaxq_f32(va, vb));
   }
   for (; i < n; ++i)
      dst[i] = a[i] > b ? a[i] : b;
}

inline void add_f32_neon_scalar_rhs(float* __restrict dst, const float* __restrict a, float b,
                                    std::size_t n) {
   std::size_t i = 0;

   const float * __restrict pa = a;
   float32x4_t vb = vdupq_n_f32(b);
   float * __restrict pd = dst;

   FUSION_CONST_ASSUME_ALIGNED(float, pa, 64);
   FUSION_ASSUME_ALIGNED(float, pd, 64);

   for (; i + kBlock <= n; i += kBlock) {
      float32x4x4_t va = vld1q_f32_x4(pa); pa += kBlock;

      va.val[0] = vaddq_f32(va.val[0], vb);
      va.val[1] = vaddq_f32(va.val[1], vb);
      va.val[2] = vaddq_f32(va.val[2], vb);
      va.val[3] = vaddq_f32(va.val[3], vb);

      vst1q_f32_x4(pd, va); pd += kBlock;

   }
   for (; i + kStep <= n; i += kStep) {
      float32x4_t va = vld1q_f32(pa); pa += kStep;
      vst1q_f32(pd, vaddq_f32(va, vb)); pd += kStep;
   }
   for (; i < n; ++i)
      *pd++ = *pa++ + b;
  }

inline void sub_f32_neon_scalar_rhs(float* __restrict dst, const float* __restrict a, float b,
                                    std::size_t n) {
   std::size_t i = 0;

   const float * __restrict pa = a;
   float32x4_t vb = vdupq_n_f32(b);
   float * __restrict pd = dst;

   FUSION_CONST_ASSUME_ALIGNED(float, pa, 64);
   FUSION_ASSUME_ALIGNED(float, pd, 64);

   for (; i + kBlock <= n; i += kBlock) {
      float32x4x4_t va = vld1q_f32_x4(pa); pa += kBlock;

      va.val[0] = vsubq_f32(va.val[0], vb);
      va.val[1] = vsubq_f32(va.val[1], vb);
      va.val[2] = vsubq_f32(va.val[2], vb);
      va.val[3] = vsubq_f32(va.val[3], vb);

      vst1q_f32_x4(pd, va); pd += kBlock;

   }
   for (; i + kStep <= n; i += kStep) {
      float32x4_t va = vld1q_f32(pa); pa += kStep;
      vst1q_f32(pd, vsubq_f32(va, vb)); pd += kStep;
   }
   for (; i < n; ++i)
      *pd++ = *pa++ - b;
  }

inline void mul_f32_neon_scalar_rhs(float* __restrict dst, const float* __restrict a, float b,
                                    std::size_t n) {
   std::size_t i = 0;

   const float * __restrict pa = a;
   float32x4_t vb = vdupq_n_f32(b);
   float * __restrict pd = dst;

   FUSION_CONST_ASSUME_ALIGNED(float, pa, 64);
   FUSION_ASSUME_ALIGNED(float, pd, 64);

   for (; i + kBlock <= n; i += kBlock) {
      float32x4x4_t va = vld1q_f32_x4(pa); pa += kBlock;

      va.val[0] = vmulq_f32(va.val[0], vb);
      va.val[1] = vmulq_f32(va.val[1], vb);
      va.val[2] = vmulq_f32(va.val[2], vb);
      va.val[3] = vmulq_f32(va.val[3], vb);

      vst1q_f32_x4(pd, va); pd += kBlock;

   }
   for (; i + kStep <= n; i += kStep) {
      float32x4_t va = vld1q_f32(pa); pa += kStep;
      vst1q_f32(pd, vmulq_f32(va, vb)); pd += kStep;
   }
   for (; i < n; ++i)
      *pd++ = *pa++ * b;
  }

inline void div_f32_neon_scalar_rhs(float* __restrict dst, const float* __restrict a, float b,
                                    std::size_t n) {

   std::size_t i = 0;

   const float * __restrict pa = a;
   float32x4_t vb = vdupq_n_f32(b);
   float * __restrict pd = dst;

   FUSION_CONST_ASSUME_ALIGNED(float, pa, 64);
   FUSION_ASSUME_ALIGNED(float, pd, 64);

   for (; i + kBlock <= n; i += kBlock) {
      float32x4x4_t va = vld1q_f32_x4(pa); pa += kBlock;

      va.val[0] = vdivq_f32(va.val[0], vb);
      va.val[1] = vdivq_f32(va.val[1], vb);
      va.val[2] = vdivq_f32(va.val[2], vb);
      va.val[3] = vdivq_f32(va.val[3], vb);

      vst1q_f32_x4(pd, va); pd += kBlock;

   }
   for (; i + kStep <= n; i += kStep) {
      float32x4_t va = vld1q_f32(pa); pa += kStep;
      vst1q_f32(pd, vdivq_f32(va, vb)); pd += kStep;
   }
   for (; i < n; ++i)
      *pd++ = *pa++ / b;
  }
#else // --------- Fallback (non-NEON builds) ---------

inline void greater_than_equal_f32_neon(float* __restrict dst, const float* __restrict a,
                                        const float* __restrict b, std::size_t n) {
   for (std::size_t i = 0; i < n; ++i)
      dst[i] = a[i] >= b[i];
}

inline void greater_than_f32_neon(float* __restrict dst, const float* __restrict a, const float* __restrict b,
                                  std::size_t n) {
   for (std::size_t i = 0; i < n; ++i)
      dst[i] = a[i] > b[i];
}

inline void greater_than_f32_neon_scalar(float* __restrict dst, const float* __restrict a, float b,
                                         std::size_t n) {
   for (std::size_t i = 0; i < n; ++i)
      dst[i] = a[i] > b;
}

inline void sum_f32_neon(float *dst, const float *a, std::size_t n) {
   float acc = 0.0f;
   for (std::size_t i = 0; i < n; ++i)
      acc += a[i];
   *dst = acc;
}
inline void sqrt_f32_neon(float* __restrict dst, const float* __restrict a, std::size_t n) {
   for (std::size_t i = 0; i < n; ++i)
      dst[i] = std::sqrt(a[i]);
}

inline void log_f32_neon(float* __restrict dst, const float* __restrict a, std::size_t n) {
   for (std::size_t i = 0; i < n; ++i)
      dst[i] = std::log(a[i]);
}

inline void exp_f32_neon(float* __restrict dst, const float* __restrict a, std::size_t n) {
   for (std::size_t i = 0; i < n; ++i)
      dst[i] = std::exp(a[i]);
}

inline void pow_f32_neon(float* __restrict__ dst, const float* __restrict__ a, const float *b,
                         std::size_t n) {
   for (std::size_t i = 0; i < n; ++i)
      dst[i] = std::pow(a[i], b[i]);
}

inline void maximum_f32_neon(float* __restrict dst, const float* __restrict a, const float* __restrict b,
                             std::size_t n) {
   for (std::size_t i = 0; i < n; ++i)
      dst[i] = a[i] > b[i] ? a[i] : b[i];
}

inline void add_f32_neon(float* __restrict dst, const float* __restrict a, const float* __restrict b,
                         std::size_t n) {
   for (std::size_t i = 0; i < n; ++i)
      dst[i] = a[i] + b[i];
}

inline void sub_f32_neon(float* __restrict dst, const float* __restrict a, const float* __restrict b,
                         std::size_t n) {
   for (std::size_t i = 0; i < n; ++i)
      dst[i] = a[i] - b[i];
}

inline void mul_f32_neon(float* __restrict dst, const float* __restrict a, const float* __restrict b,
                         std::size_t n) {
   for (std::size_t i = 0; i < n; ++i)
      dst[i] = a[i] * b[i];
}

inline void div_f32_neon(float* __restrict dst, const float* __restrict a, const float* __restrict b,
                         std::size_t n) {
   for (std::size_t i = 0; i < n; ++i)
      dst[i] = a[i] / b[i];
}

inline void greater_than_equal_f32_neon_scalar(float* __restrict dst, const float* __restrict a,
                                               float b, std::size_t n) {
   for (std::size_t i = 0; i < n; ++i)
      dst[i] = a[i] >= b;
}

inline void pow_f32_neon_scalar_rhs(float* __restrict dst, const float* __restrict a, float b,
                                    std::size_t n) {
   for (std::size_t i = 0; i < n; ++i)
      dst[i] = std::pow(a[i], b);
}

inline void add_f32_neon_scalar_rhs(float* __restrict dst, const float* __restrict a, float b,
                                    std::size_t n) {
   for (std::size_t i = 0; i < n; ++i)
      dst[i] = a[i] + b;
}

inline void sub_f32_neon_scalar_rhs(float* __restrict dst, const float* __restrict a, float b,
                                    std::size_t n) {
   for (std::size_t i = 0; i < n; ++i)
      dst[i] = a[i] - b;
}

inline void mul_f32_neon_scalar_rhs(float* __restrict dst, const float* __restrict a, float b,
                                    std::size_t n) {
   for (std::size_t i = 0; i < n; ++i)
      dst[i] = a[i] * b;
}

inline void div_f32_neon_scalar_rhs(float* __restrict dst, const float* __restrict a, float b,
                                    std::size_t n) {
   for (std::size_t i = 0; i < n; ++i)
      dst[i] = a[i] / b;
}

inline void maximum_f32_neon_scalar_rhs(float* __restrict dst, const float* __restrict a,
                                        const float b, std::size_t n) {
   for (std::size_t i = 0; i < n; ++i)
      dst[i] = a[i] > b ? a[i] : b;
}

#endif
} // namespace simd

#endif // FUSION_VEC128_NEON_H
