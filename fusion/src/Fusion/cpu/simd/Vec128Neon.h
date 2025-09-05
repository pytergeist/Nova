#pragma once
#ifndef FUSION_VEC128_NEON_H
#define FUSION_VEC128_NEON_H

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <sleef.h>

#if defined(FUSION_ENABLE_NEON) &&                                             \
    (defined(__ARM_NEON) || defined(__ARM_NEON__))
#include <arm_neon.h>
#endif

namespace simd {
static constexpr std::size_t kNeonVectorBytes = 16;

#if defined(FUSION_ENABLE_NEON) &&                                             \
    (defined(__ARM_NEON) || defined(__ARM_NEON__))
// =========================
// Core contiguous kernels
// =========================
// All assume: a, b, dst are contiguous float buffers of length n.

inline void sum_f32_neon(float *dst, const float *a, std::size_t n) {
#if defined(FUSION_ENABLE_NEON) &&                                             \
    (defined(__ARM_NEON) || defined(__ARM_NEON__))
  std::size_t i = 0;
  float32x4_t acc0 = vdupq_n_f32(0.0f);
  float32x4_t acc1 = vdupq_n_f32(0.0f);
  float32x4_t acc2 = vdupq_n_f32(0.0f);
  float32x4_t acc3 = vdupq_n_f32(0.0f);

  for (; i + 16 <= n; i += 16) {
    float32x4_t v0 = vld1q_f32(a + i + 0);
    float32x4_t v1 = vld1q_f32(a + i + 4);
    float32x4_t v2 = vld1q_f32(a + i + 8);
    float32x4_t v3 = vld1q_f32(a + i + 12);
    acc0 = vaddq_f32(acc0, v0);
    acc1 = vaddq_f32(acc1, v1);
    acc2 = vaddq_f32(acc2, v2);
    acc3 = vaddq_f32(acc3, v3);
  }

  float32x4_t acc = vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3));

  for (; i + 4 <= n; i += 4) {
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

inline void sqrt_f32_neon(float *dst, const float *a, std::size_t n) {
  std::size_t i = 0;

  for (; i + 16 <= n; i += 16) {
    float32x4_t a0 = vld1q_f32(a + i + 0);
    float32x4_t a1 = vld1q_f32(a + i + 4);
    float32x4_t a2 = vld1q_f32(a + i + 8);
    float32x4_t a3 = vld1q_f32(a + i + 12);

    vst1q_f32(dst + i + 0, Sleef_sqrtf4_u05(a0));
    vst1q_f32(dst + i + 4, Sleef_sqrtf4_u05(a1));
    vst1q_f32(dst + i + 8, Sleef_sqrtf4_u05(a2));
    vst1q_f32(dst + i + 12, Sleef_sqrtf4_u05(a3));
  }
  for (; i + 4 <= n; i += 4) {
    float32x4_t va = vld1q_f32(a + i);
    vst1q_f32(dst + i, Sleef_sqrtf4_u05(va));
  }
  for (; i < n; ++i)
    dst[i] = std::sqrt(a[i]);
}

inline void exp_f32_neon(float *dst, const float *a, std::size_t n) {
  std::size_t i = 0;

  for (; i + 16 <= n; i += 16) {
    float32x4_t a0 = vld1q_f32(a + i + 0);
    float32x4_t a1 = vld1q_f32(a + i + 4);
    float32x4_t a2 = vld1q_f32(a + i + 8);
    float32x4_t a3 = vld1q_f32(a + i + 12);

    vst1q_f32(dst + i + 0, Sleef_expf4_u10(a0));
    vst1q_f32(dst + i + 4, Sleef_expf4_u10(a1));
    vst1q_f32(dst + i + 8, Sleef_expf4_u10(a2));
    vst1q_f32(dst + i + 12, Sleef_expf4_u10(a3));
  }
  for (; i + 4 <= n; i += 4) {
    float32x4_t va = vld1q_f32(a + i);
    vst1q_f32(dst + i, Sleef_expf4_u10(va));
  }
  for (; i < n; ++i)
    dst[i] = std::exp(a[i]);
}

inline void log_f32_neon(float *dst, const float *a, std::size_t n) {
  std::size_t i = 0;

  for (; i + 16 <= n; i += 16) {
    float32x4_t a0 = vld1q_f32(a + i + 0);
    float32x4_t a1 = vld1q_f32(a + i + 4);
    float32x4_t a2 = vld1q_f32(a + i + 8);
    float32x4_t a3 = vld1q_f32(a + i + 12);

    vst1q_f32(dst + i + 0, Sleef_logf4_u10(a0));
    vst1q_f32(dst + i + 4, Sleef_logf4_u10(a1));
    vst1q_f32(dst + i + 8, Sleef_logf4_u10(a2));
    vst1q_f32(dst + i + 12, Sleef_logf4_u10(a3));
  }
  for (; i + 4 <= n; i += 4) {
    float32x4_t va = vld1q_f32(a + i);
    vst1q_f32(dst + i, Sleef_logf4_u10(va));
  }
  for (; i < n; ++i)
    dst[i] = std::log(a[i]);
}

inline void pow_f32_neon(float *dst, const float *a, const float *b,
                         std::size_t n) {
  std::size_t i = 0;

  for (; i + 16 <= n; i += 16) {
    float32x4_t a0 = vld1q_f32(a + i + 0);
    float32x4_t a1 = vld1q_f32(a + i + 4);
    float32x4_t a2 = vld1q_f32(a + i + 8);
    float32x4_t a3 = vld1q_f32(a + i + 12);

    float32x4_t b0 = vld1q_f32(b + i + 0);
    float32x4_t b1 = vld1q_f32(b + i + 4);
    float32x4_t b2 = vld1q_f32(b + i + 8);

    float32x4_t b3 = vld1q_f32(b + i + 12);

    vst1q_f32(dst + i + 0, Sleef_powf4_u10(a0, b0));
    vst1q_f32(dst + i + 4, Sleef_powf4_u10(a1, b1));
    vst1q_f32(dst + i + 8, Sleef_powf4_u10(a2, b2));
    vst1q_f32(dst + i + 12, Sleef_powf4_u10(a3, b3));
  }
  for (; i + 4 <= n; i += 4) {
    float32x4_t va = vld1q_f32(a + i);
    float32x4_t vb = vld1q_f32(b + i);
    vst1q_f32(dst + i, Sleef_powf4_u10(va, vb));
  }
  for (; i < n; ++i)
    dst[i] = std::pow(a[i], b[i]);
}

inline void maximum_f32_neon(float *dst, const float *a, const float *b,
                             std::size_t n) {
  std::size_t i = 0;

  for (; i + 16 <= n; i += 16) {
    float32x4_t a0 = vld1q_f32(a + i + 0);
    float32x4_t a1 = vld1q_f32(a + i + 4);
    float32x4_t a2 = vld1q_f32(a + i + 8);
    float32x4_t a3 = vld1q_f32(a + i + 12);

    float32x4_t b0 = vld1q_f32(b + i + 0);
    float32x4_t b1 = vld1q_f32(b + i + 4);
    float32x4_t b2 = vld1q_f32(b + i + 8);
    float32x4_t b3 = vld1q_f32(b + i + 12);

    vst1q_f32(dst + i + 0,  vmaxq_f32(a0, b0));
    vst1q_f32(dst + i + 4,  vmaxq_f32(a1, b1));
    vst1q_f32(dst + i + 8,  vmaxq_f32(a2, b2));
    vst1q_f32(dst + i + 12, vmaxq_f32(a3, b3));
  }
  for (; i + 4 <= n; i += 4) {
    float32x4_t va = vld1q_f32(a + i);
    float32x4_t vb = vld1q_f32(b + i);
    vst1q_f32(dst + i, vmaxq_f32(va, vb));
  }
  for (; i < n; ++i)
    dst[i] = a[i] > b[i] ? a[i] : b[i];
}

inline void greater_than_f32_neon(float *dst, const float *a, const float *b,
                                  std::size_t n) {
  std::size_t i = 0;

  for (; i + 16 <= n; i += 16) {
    float32x4_t a0 = vld1q_f32(a + i + 0);
    float32x4_t a1 = vld1q_f32(a + i + 4);
    float32x4_t a2 = vld1q_f32(a + i + 8);
    float32x4_t a3 = vld1q_f32(a + i + 12);

    float32x4_t b0 = vld1q_f32(b + i + 0);
    float32x4_t b1 = vld1q_f32(b + i + 4);
    float32x4_t b2 = vld1q_f32(b + i + 8);
    float32x4_t b3 = vld1q_f32(b + i + 12);

    uint32x4_t mask0 = vcgtq_f32(a0, b0);
    vst1q_f32(dst + i + 0,
              vbslq_f32(mask0, vdupq_n_f32(1.0f), vdupq_n_f32(0.0f)));
    uint32x4_t mask1 = vcgtq_f32(a1, b1);
    vst1q_f32(dst + i + 4,
              vbslq_f32(mask1, vdupq_n_f32(1.0f), vdupq_n_f32(0.0f)));
    uint32x4_t mask2 = vcgtq_f32(a2, b2);
    vst1q_f32(dst + i + 8,
              vbslq_f32(mask2, vdupq_n_f32(1.0f), vdupq_n_f32(0.0f)));
    uint32x4_t mask3 = vcgtq_f32(a3, b3);
    vst1q_f32(dst + i + 12,
              vbslq_f32(mask3, vdupq_n_f32(1.0f), vdupq_n_f32(0.0f)));
  }
  for (; i + 4 <= n; i += 4) {
    float32x4_t va = vld1q_f32(a + i);
    float32x4_t vb = vld1q_f32(b + i);
    uint32x4_t mask_va = vcgtq_f32(va, vb);
    vst1q_f32(dst + i,
              vbslq_f32(mask_va, vdupq_n_f32(1.0f), vdupq_n_f32(0.0f)));
  }
  for (; i < n; ++i)
    dst[i] = a[i] > b[i];
}

inline void greater_than_equal_f32_neon(float *dst, const float *a,
                                        const float *b, std::size_t n) {
  std::size_t i = 0;

  for (; i + 16 <= n; i += 16) {
    float32x4_t a0 = vld1q_f32(a + i + 0);
    float32x4_t a1 = vld1q_f32(a + i + 4);
    float32x4_t a2 = vld1q_f32(a + i + 8);
    float32x4_t a3 = vld1q_f32(a + i + 12);

    float32x4_t b0 = vld1q_f32(b + i + 0);
    float32x4_t b1 = vld1q_f32(b + i + 4);
    float32x4_t b2 = vld1q_f32(b + i + 8);
    float32x4_t b3 = vld1q_f32(b + i + 12);

    uint32x4_t mask0 = vcgeq_f32(a0, b0);
    vst1q_f32(dst + i + 0,
              vbslq_f32(mask0, vdupq_n_f32(1.0f), vdupq_n_f32(0.0f)));
    uint32x4_t mask1 = vcgeq_f32(a1, b1);
    vst1q_f32(dst + i + 4,
              vbslq_f32(mask1, vdupq_n_f32(1.0f), vdupq_n_f32(0.0f)));
    uint32x4_t mask2 = vcgeq_f32(a2, b2);
    vst1q_f32(dst + i + 8,
              vbslq_f32(mask2, vdupq_n_f32(1.0f), vdupq_n_f32(0.0f)));
    uint32x4_t mask3 = vcgeq_f32(a3, b3);
    vst1q_f32(dst + i + 12,
              vbslq_f32(mask3, vdupq_n_f32(1.0f), vdupq_n_f32(0.0f)));
  }
  for (; i + 4 <= n; i += 4) {
    float32x4_t va = vld1q_f32(a + i);
    float32x4_t vb = vld1q_f32(b + i);
    uint32x4_t mask_va = vcgeq_f32(va, vb);
    vst1q_f32(dst + i,
              vbslq_f32(mask_va, vdupq_n_f32(1.0f), vdupq_n_f32(0.0f)));
  }
  for (; i < n; ++i)
    dst[i] = a[i] >= b[i];
}

inline void add_f32_neon(float *dst, const float *a, const float *b,
                         std::size_t n) {
  std::size_t i = 0;

  for (; i + 16 <= n; i += 16) {
    float32x4_t a0 = vld1q_f32(a + i + 0);
    float32x4_t a1 = vld1q_f32(a + i + 4);
    float32x4_t a2 = vld1q_f32(a + i + 8);
    float32x4_t a3 = vld1q_f32(a + i + 12);

    float32x4_t b0 = vld1q_f32(b + i + 0);
    float32x4_t b1 = vld1q_f32(b + i + 4);
    float32x4_t b2 = vld1q_f32(b + i + 8);

    float32x4_t b3 = vld1q_f32(b + i + 12);

    vst1q_f32(dst + i + 0, vaddq_f32(a0, b0));
    vst1q_f32(dst + i + 4, vaddq_f32(a1, b1));
    vst1q_f32(dst + i + 8, vaddq_f32(a2, b2));
    vst1q_f32(dst + i + 12, vaddq_f32(a3, b3));
  }
  for (; i + 4 <= n; i += 4) {
    float32x4_t va = vld1q_f32(a + i);
    float32x4_t vb = vld1q_f32(b + i);
    vst1q_f32(dst + i, vaddq_f32(va, vb));
  }
  for (; i < n; ++i)
    dst[i] = a[i] + b[i];
}

inline void sub_f32_neon(float *dst, const float *a, const float *b,
                         std::size_t n) {
  std::size_t i = 0;
  for (; i + 16 <= n; i += 16) {
    float32x4_t a0 = vld1q_f32(a + i + 0);
    float32x4_t a1 = vld1q_f32(a + i + 4);
    float32x4_t a2 = vld1q_f32(a + i + 8);
    float32x4_t a3 = vld1q_f32(a + i + 12);

    float32x4_t b0 = vld1q_f32(b + i + 0);
    float32x4_t b1 = vld1q_f32(b + i + 4);
    float32x4_t b2 = vld1q_f32(b + i + 8);
    float32x4_t b3 = vld1q_f32(b + i + 12);

    vst1q_f32(dst + i + 0, vsubq_f32(a0, b0));
    vst1q_f32(dst + i + 4, vsubq_f32(a1, b1));
    vst1q_f32(dst + i + 8, vsubq_f32(a2, b2));
    vst1q_f32(dst + i + 12, vsubq_f32(a3, b3));
  }
  for (; i + 4 <= n; i += 4) {
    float32x4_t va = vld1q_f32(a + i);
    float32x4_t vb = vld1q_f32(b + i);
    vst1q_f32(dst + i, vsubq_f32(va, vb));
  }
  for (; i < n; ++i)
    dst[i] = a[i] - b[i];
}

inline void mul_f32_neon(float *dst, const float *a, const float *b,
                         std::size_t n) {
  std::size_t i = 0;
  for (; i + 16 <= n; i += 16) {
    float32x4_t a0 = vld1q_f32(a + i + 0);
    float32x4_t a1 = vld1q_f32(a + i + 4);
    float32x4_t a2 = vld1q_f32(a + i + 8);
    float32x4_t a3 = vld1q_f32(a + i + 12);

    float32x4_t b0 = vld1q_f32(b + i + 0);
    float32x4_t b1 = vld1q_f32(b + i + 4);
    float32x4_t b2 = vld1q_f32(b + i + 8);
    float32x4_t b3 = vld1q_f32(b + i + 12);

    vst1q_f32(dst + i + 0, vmulq_f32(a0, b0));
    vst1q_f32(dst + i + 4, vmulq_f32(a1, b1));
    vst1q_f32(dst + i + 8, vmulq_f32(a2, b2));
    vst1q_f32(dst + i + 12, vmulq_f32(a3, b3));
  }
  for (; i + 4 <= n; i += 4) {
    float32x4_t va = vld1q_f32(a + i);
    float32x4_t vb = vld1q_f32(b + i);
    vst1q_f32(dst + i, vmulq_f32(va, vb));
  }
  for (; i < n; ++i)
    dst[i] = a[i] * b[i];
}

inline void div_f32_neon(float *dst, const float *a, const float *b,
                         std::size_t n) {
  std::size_t i = 0;
  for (; i + 16 <= n; i += 16) {
    float32x4_t a0 = vld1q_f32(a + i + 0);
    float32x4_t a1 = vld1q_f32(a + i + 4);
    float32x4_t a2 = vld1q_f32(a + i + 8);
    float32x4_t a3 = vld1q_f32(a + i + 12);

    float32x4_t b0 = vld1q_f32(b + i + 0);
    float32x4_t b1 = vld1q_f32(b + i + 4);
    float32x4_t b2 = vld1q_f32(b + i + 8);
    float32x4_t b3 = vld1q_f32(b + i + 12);

    vst1q_f32(dst + i + 0, vdivq_f32(a0, b0));
    vst1q_f32(dst + i + 4, vdivq_f32(a1, b1));
    vst1q_f32(dst + i + 8, vdivq_f32(a2, b2));
    vst1q_f32(dst + i + 12, vdivq_f32(a3, b3));
  }
  for (; i + 4 <= n; i += 4) {
    float32x4_t va = vld1q_f32(a + i);
    float32x4_t vb = vld1q_f32(b + i);
    vst1q_f32(dst + i, vdivq_f32(va, vb));
  }
  for (; i < n; ++i)
    dst[i] = a[i] / b[i];
}

// =========================
// Scalar-RHS wrappers
// ========================= // TODO: Rename from RHS
// Useful when your inner dim sees stride==0 for RHS/LHS (broadcast scalar).
// If LHS is the scalar instead, you can either add "scalar_lhs" variants
// or just swap operands in the caller for commutative ops.

inline void greater_than_f32_neon_scalar(float *dst, const float *a,
                                         const float b, std::size_t n) {
  std::size_t i = 0;

  for (; i + 16 <= n; i += 16) {
    float32x4_t a0 = vld1q_f32(a + i + 0);
    float32x4_t a1 = vld1q_f32(a + i + 4);
    float32x4_t a2 = vld1q_f32(a + i + 8);
    float32x4_t a3 = vld1q_f32(a + i + 12);

    float32x4_t vb = vdupq_n_f32(b);
    uint32x4_t mask0 = vcgtq_f32(a0, vb);
    vst1q_f32(dst + i + 0,
              vbslq_f32(mask0, vdupq_n_f32(1.0f), vdupq_n_f32(0.0f)));
    uint32x4_t mask1 = vcgtq_f32(a1, vb);
    vst1q_f32(dst + i + 4,
              vbslq_f32(mask1, vdupq_n_f32(1.0f), vdupq_n_f32(0.0f)));
    uint32x4_t mask2 = vcgtq_f32(a2, vb);
    vst1q_f32(dst + i + 8,
              vbslq_f32(mask2, vdupq_n_f32(1.0f), vdupq_n_f32(0.0f)));
    uint32x4_t mask3 = vcgtq_f32(a3, vb);
    vst1q_f32(dst + i + 12,
              vbslq_f32(mask3, vdupq_n_f32(1.0f), vdupq_n_f32(0.0f)));
  }
  for (; i + 4 <= n; i += 4) {
    float32x4_t va = vld1q_f32(a + i);
    float32x4_t vb = vdupq_n_f32(b);
    uint32x4_t mask_va = vcgtq_f32(va, vb);
    vst1q_f32(dst + i,
              vbslq_f32(mask_va, vdupq_n_f32(1.0f), vdupq_n_f32(0.0f)));
  }
  for (; i < n; ++i)
    dst[i] = a[i] > b;
}

inline void greater_than_equal_f32_neon_scalar(float *dst, const float *a,
                                               const float b, std::size_t n) {
  std::size_t i = 0;

  for (; i + 16 <= n; i += 16) {
    float32x4_t a0 = vld1q_f32(a + i + 0);
    float32x4_t a1 = vld1q_f32(a + i + 4);
    float32x4_t a2 = vld1q_f32(a + i + 8);
    float32x4_t a3 = vld1q_f32(a + i + 12);

    float32x4_t vb = vdupq_n_f32(b);
    uint32x4_t mask0 = vcgeq_f32(a0, vb);
    vst1q_f32(dst + i + 0,
              vbslq_f32(mask0, vdupq_n_f32(1.0f), vdupq_n_f32(0.0f)));
    uint32x4_t mask1 = vcgeq_f32(a1, vb);
    vst1q_f32(dst + i + 4,
              vbslq_f32(mask1, vdupq_n_f32(1.0f), vdupq_n_f32(0.0f)));
    uint32x4_t mask2 = vcgeq_f32(a2, vb);
    vst1q_f32(dst + i + 8,
              vbslq_f32(mask2, vdupq_n_f32(1.0f), vdupq_n_f32(0.0f)));
    uint32x4_t mask3 = vcgeq_f32(a3, vb);
    vst1q_f32(dst + i + 12,
              vbslq_f32(mask3, vdupq_n_f32(1.0f), vdupq_n_f32(0.0f)));
  }
  for (; i + 4 <= n; i += 4) {
    float32x4_t va = vld1q_f32(a + i);
    float32x4_t vb = vdupq_n_f32(b);
    uint32x4_t mask_va = vcgeq_f32(va, vb);
    vst1q_f32(dst + i,
              vbslq_f32(mask_va, vdupq_n_f32(1.0f), vdupq_n_f32(0.0f)));
  }
  for (; i < n; ++i)
    dst[i] = a[i] >= b;
}

inline void pow_f32_neon_scalar_rhs(float *dst, const float *a, const float b,
                                    std::size_t n) {
  float32x4_t vb = vdupq_n_f32(b);
  std::size_t i = 0;
  for (; i + 16 <= n; i += 16) {
    float32x4_t a0 = vld1q_f32(a + i + 0);
    float32x4_t a1 = vld1q_f32(a + i + 4);
    float32x4_t a2 = vld1q_f32(a + i + 8);
    float32x4_t a3 = vld1q_f32(a + i + 12);

    vst1q_f32(dst + i + 0, Sleef_powf4_u10(a0, vb));
    vst1q_f32(dst + i + 4, Sleef_powf4_u10(a1, vb));
    vst1q_f32(dst + i + 8, Sleef_powf4_u10(a2, vb));
    vst1q_f32(dst + i + 12, Sleef_powf4_u10(a3, vb));
  }
  for (; i + 4 <= n; i += 4) {
    float32x4_t va = vld1q_f32(a + i);
    vst1q_f32(dst + i, Sleef_powf4_u10(va, vb));
  }
  for (; i < n; ++i)
    dst[i] = std::pow(a[i], b);
}

inline void maximum_f32_neon_scalar_rhs(float *dst, const float *a, float b,
                                        std::size_t n) {
  float32x4_t vb = vdupq_n_f32(b);
  std::size_t i = 0;
  for (; i + 16 <= n; i += 16) {
    float32x4_t a0 = vld1q_f32(a + i + 0);
    float32x4_t a1 = vld1q_f32(a + i + 4);
    float32x4_t a2 = vld1q_f32(a + i + 8);
    float32x4_t a3 = vld1q_f32(a + i + 12);
    vst1q_f32(dst + i + 0,  vmaxq_f32(a0, vb));
    vst1q_f32(dst + i + 4,  vmaxq_f32(a1, vb));
    vst1q_f32(dst + i + 8,  vmaxq_f32(a2, vb));
    vst1q_f32(dst + i + 12, vmaxq_f32(a3, vb));
  }
  for (; i + 4 <= n; i += 4) {
    float32x4_t va = vld1q_f32(a + i);
    vst1q_f32(dst + i, vmaxq_f32(va, vb));
  }
  for (; i < n; ++i)
    dst[i] = a[i] > b ? a[i] : b;
}

inline void add_f32_neon_scalar_rhs(float *dst, const float *a, float b,
                                    std::size_t n) {
  float32x4_t vb = vdupq_n_f32(b);
  std::size_t i = 0;
  for (; i + 16 <= n; i += 16) {
    float32x4_t a0 = vld1q_f32(a + i + 0);
    float32x4_t a1 = vld1q_f32(a + i + 4);
    float32x4_t a2 = vld1q_f32(a + i + 8);
    float32x4_t a3 = vld1q_f32(a + i + 12);
    vst1q_f32(dst + i + 0, vaddq_f32(a0, vb));
    vst1q_f32(dst + i + 4, vaddq_f32(a1, vb));
    vst1q_f32(dst + i + 8, vaddq_f32(a2, vb));
    vst1q_f32(dst + i + 12, vaddq_f32(a3, vb));
  }
  for (; i + 4 <= n; i += 4) {
    float32x4_t va = vld1q_f32(a + i);
    vst1q_f32(dst + i, vaddq_f32(va, vb));
  }
  for (; i < n; ++i)
    dst[i] = a[i] + b;
}

inline void sub_f32_neon_scalar_rhs(float *dst, const float *a, float b,
                                    std::size_t n) {
  float32x4_t vb = vdupq_n_f32(b);
  std::size_t i = 0;
  for (; i + 16 <= n; i += 16) {
    float32x4_t a0 = vld1q_f32(a + i + 0);
    float32x4_t a1 = vld1q_f32(a + i + 4);
    float32x4_t a2 = vld1q_f32(a + i + 8);
    float32x4_t a3 = vld1q_f32(a + i + 12);
    vst1q_f32(dst + i + 0, vsubq_f32(a0, vb));
    vst1q_f32(dst + i + 4, vsubq_f32(a1, vb));
    vst1q_f32(dst + i + 8, vsubq_f32(a2, vb));
    vst1q_f32(dst + i + 12, vsubq_f32(a3, vb));
  }
  for (; i + 4 <= n; i += 4) {
    float32x4_t va = vld1q_f32(a + i);
    vst1q_f32(dst + i, vsubq_f32(va, vb));
  }
  for (; i < n; ++i)
    dst[i] = a[i] - b;
}

inline void mul_f32_neon_scalar_rhs(float *dst, const float *a, float b,
                                    std::size_t n) {
  float32x4_t vb = vdupq_n_f32(b);
  std::size_t i = 0;
  for (; i + 16 <= n; i += 16) {
    float32x4_t a0 = vld1q_f32(a + i + 0);
    float32x4_t a1 = vld1q_f32(a + i + 4);
    float32x4_t a2 = vld1q_f32(a + i + 8);
    float32x4_t a3 = vld1q_f32(a + i + 12);
    vst1q_f32(dst + i + 0, vmulq_f32(a0, vb));
    vst1q_f32(dst + i + 4, vmulq_f32(a1, vb));
    vst1q_f32(dst + i + 8, vmulq_f32(a2, vb));
    vst1q_f32(dst + i + 12, vmulq_f32(a3, vb));
  }
  for (; i + 4 <= n; i += 4) {
    float32x4_t va = vld1q_f32(a + i);
    vst1q_f32(dst + i, vmulq_f32(va, vb));
  }
  for (; i < n; ++i)
    dst[i] = a[i] * b;
}

inline void div_f32_neon_scalar_rhs(float *dst, const float *a, float b,
                                    std::size_t n) {
  float32x4_t vb = vdupq_n_f32(b);
  std::size_t i = 0;
  for (; i + 16 <= n; i += 16) {
    float32x4_t a0 = vld1q_f32(a + i + 0);
    float32x4_t a1 = vld1q_f32(a + i + 4);
    float32x4_t a2 = vld1q_f32(a + i + 8);
    float32x4_t a3 = vld1q_f32(a + i + 12);
    vst1q_f32(dst + i + 0, vdivq_f32(a0, vb));
    vst1q_f32(dst + i + 4, vdivq_f32(a1, vb));
    vst1q_f32(dst + i + 8, vdivq_f32(a2, vb));
    vst1q_f32(dst + i + 12, vdivq_f32(a3, vb));
  }
  for (; i + 4 <= n; i += 4) {
    float32x4_t va = vld1q_f32(a + i);
    vst1q_f32(dst + i, vdivq_f32(va, vb));
  }
  for (; i < n; ++i)
    dst[i] = a[i] / b;
}

#else // --------- Fallback (non-NEON builds) ---------

inline void greater_than_equal_f32_neon(float *dst, const float *a,
                                        const float *b, std::size_t n) {
  for (std::size_t i = 0; i < n; ++i)
    dst[i] = a[i] >= b[i];
}

inline void greater_than_f32_neon(float *dst, const float *a, const float *b,
                                  std::size_t n) {
  for (std::size_t i = 0; i < n; ++i)
    dst[i] = a[i] > b[i];
}

  inline void  greater_than_f32_neon_scalar(float *dst, const float *a, float b,
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
inline void sqrt_f32_neon(float *dst, const float *a, std::size_t n) {
  for (std::size_t i = 0; i < n; ++i)
    dst[i] = std::sqrt(a[i]);
}

inline void log_f32_neon(float *dst, const float *a, std::size_t n) {
  for (std::size_t i = 0; i < n; ++i)
    dst[i] = std::log(a[i]);
}

inline void exp_f32_neon(float *dst, const float *a, std::size_t n) {
  for (std::size_t i = 0; i < n; ++i)
    dst[i] = std::exp(a[i]);
}

inline void pow_f32_neon(float *dst, const float *a, const float *b,
                         std::size_t n) {
  for (std::size_t i = 0; i < n; ++i)
    dst[i] = std::pow(a[i], b[i]);
}

inline void maximum_f32_neon(float *dst, const float *a, const float *b,
                             std::size_t n) {
  for (std::size_t i = 0; i < n; ++i)
    dst[i] = a[i] > b[i] ? a[i] : b[i];
}

inline void add_f32_neon(float *dst, const float *a, const float *b,
                         std::size_t n) {
  for (std::size_t i = 0; i < n; ++i)
    dst[i] = a[i] + b[i];
}

inline void sub_f32_neon(float *dst, const float *a, const float *b,
                         std::size_t n) {
  for (std::size_t i = 0; i < n; ++i)
    dst[i] = a[i] - b[i];
}

inline void mul_f32_neon(float *dst, const float *a, const float *b,
                         std::size_t n) {
  for (std::size_t i = 0; i < n; ++i)
    dst[i] = a[i] * b[i];
}

inline void div_f32_neon(float *dst, const float *a, const float *b,
                         std::size_t n) {
  for (std::size_t i = 0; i < n; ++i)
    dst[i] = a[i] / b[i];
}

inline void greater_than_equal_f32_neon_scalar(float *dst, const float *a,
                                               float b, std::size_t n) {
  for (std::size_t i = 0; i < n; ++i)
    dst[i] = a[i] >= b;
}

inline void pow_f32_neon_scalar_rhs(float *dst, const float *a, float b,
                                    std::size_t n) {
  for (std::size_t i = 0; i < n; ++i)
    dst[i] = std::pow(a[i], b);
}

inline void add_f32_neon_scalar_rhs(float *dst, const float *a, float b,
                                    std::size_t n) {
  for (std::size_t i = 0; i < n; ++i)
    dst[i] = a[i] + b;
}

inline void sub_f32_neon_scalar_rhs(float *dst, const float *a, float b,
                                    std::size_t n) {
  for (std::size_t i = 0; i < n; ++i)
    dst[i] = a[i] - b;
}

inline void mul_f32_neon_scalar_rhs(float *dst, const float *a, float b,
                                    std::size_t n) {
  for (std::size_t i = 0; i < n; ++i)
    dst[i] = a[i] * b;
}

inline void div_f32_neon_scalar_rhs(float *dst, const float *a, float b,
                                    std::size_t n) {
  for (std::size_t i = 0; i < n; ++i)
    dst[i] = a[i] / b;
}

inline void maximum_f32_neon_scalar_rhs(float *dst, const float *a,
                                        const float b, std::size_t n) {
  for (std::size_t i = 0; i < n; ++i)
    dst[i] = a[i] > b ? a[i] : b;
}

#endif
} // namespace simd

#endif // FUSION_VEC128_NEON_H
