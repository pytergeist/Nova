#pragma once
#ifndef FUSION_VEC128_NEON_H
#define FUSION_VEC128_NEON_H

#include <cstddef>
#include <cstdint>

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
  #include <arm_neon.h>
#endif

namespace simd {

static constexpr std::size_t kNeonVectorBytes = 16;

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
// =========================
// Core contiguous kernels
// =========================
// All assume: a, b, dst are contiguous float buffers of length n.

inline void add_f32_neon(float* dst, const float* a, const float* b, std::size_t n) {
    std::size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        float32x4_t a0 = vld1q_f32(a + i +  0);
        float32x4_t a1 = vld1q_f32(a + i +  4);
        float32x4_t a2 = vld1q_f32(a + i +  8);
        float32x4_t a3 = vld1q_f32(a + i + 12);

        float32x4_t b0 = vld1q_f32(b + i +  0);
        float32x4_t b1 = vld1q_f32(b + i +  4);
        float32x4_t b2 = vld1q_f32(b + i +  8);
        float32x4_t b3 = vld1q_f32(b + i + 12);

        vst1q_f32(dst + i +  0, vaddq_f32(a0, b0));
        vst1q_f32(dst + i +  4, vaddq_f32(a1, b1));
        vst1q_f32(dst + i +  8, vaddq_f32(a2, b2));
        vst1q_f32(dst + i + 12, vaddq_f32(a3, b3));
    }
    for (; i + 4 <= n; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        vst1q_f32(dst + i, vaddq_f32(va, vb));
    }
    for (; i < n; ++i) dst[i] = a[i] + b[i];
}

inline void sub_f32_neon(float* dst, const float* a, const float* b, std::size_t n) {
    std::size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        float32x4_t a0 = vld1q_f32(a + i +  0);
        float32x4_t a1 = vld1q_f32(a + i +  4);
        float32x4_t a2 = vld1q_f32(a + i +  8);
        float32x4_t a3 = vld1q_f32(a + i + 12);

        float32x4_t b0 = vld1q_f32(b + i +  0);
        float32x4_t b1 = vld1q_f32(b + i +  4);
        float32x4_t b2 = vld1q_f32(b + i +  8);
        float32x4_t b3 = vld1q_f32(b + i + 12);

        vst1q_f32(dst + i +  0, vsubq_f32(a0, b0));
        vst1q_f32(dst + i +  4, vsubq_f32(a1, b1));
        vst1q_f32(dst + i +  8, vsubq_f32(a2, b2));
        vst1q_f32(dst + i + 12, vsubq_f32(a3, b3));
    }
    for (; i + 4 <= n; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        vst1q_f32(dst + i, vsubq_f32(va, vb));
    }
    for (; i < n; ++i) dst[i] = a[i] - b[i];
}

inline void mul_f32_neon(float* dst, const float* a, const float* b, std::size_t n) {
    std::size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        float32x4_t a0 = vld1q_f32(a + i +  0);
        float32x4_t a1 = vld1q_f32(a + i +  4);
        float32x4_t a2 = vld1q_f32(a + i +  8);
        float32x4_t a3 = vld1q_f32(a + i + 12);

        float32x4_t b0 = vld1q_f32(b + i +  0);
        float32x4_t b1 = vld1q_f32(b + i +  4);
        float32x4_t b2 = vld1q_f32(b + i +  8);
        float32x4_t b3 = vld1q_f32(b + i + 12);

        vst1q_f32(dst + i +  0, vmulq_f32(a0, b0));
        vst1q_f32(dst + i +  4, vmulq_f32(a1, b1));
        vst1q_f32(dst + i +  8, vmulq_f32(a2, b2));
        vst1q_f32(dst + i + 12, vmulq_f32(a3, b3));
    }
    for (; i + 4 <= n; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        vst1q_f32(dst + i, vmulq_f32(va, vb));
    }
    for (; i < n; ++i) dst[i] = a[i] * b[i];
}


inline void div_f32_neon(float* dst, const float* a, const float* b, std::size_t n) {
    std::size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        float32x4_t a0 = vld1q_f32(a + i +  0);
        float32x4_t a1 = vld1q_f32(a + i +  4);
        float32x4_t a2 = vld1q_f32(a + i +  8);
        float32x4_t a3 = vld1q_f32(a + i + 12);

        float32x4_t b0 = vld1q_f32(b + i +  0);
        float32x4_t b1 = vld1q_f32(b + i +  4);
        float32x4_t b2 = vld1q_f32(b + i +  8);
        float32x4_t b3 = vld1q_f32(b + i + 12);

        vst1q_f32(dst + i +  0, vdivq_f32(a0, b0));
        vst1q_f32(dst + i +  4, vdivq_f32(a1, b1));
        vst1q_f32(dst + i +  8, vdivq_f32(a2, b2));
        vst1q_f32(dst + i + 12, vdivq_f32(a3, b3));
    }
    for (; i + 4 <= n; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        vst1q_f32(dst + i, vdivq_f32(va, vb));
    }
    for (; i < n; ++i) dst[i] = a[i] / b[i];
}

// =========================
// Scalar-RHS wrappers
// =========================
// Useful when your inner dim sees stride==0 for RHS (broadcasted scalar).
// If LHS is the scalar instead, you can either add "scalar_lhs" variants
// or just swap operands in the caller for commutative ops.

inline void add_f32_neon_scalar_rhs(float* dst, const float* a, float b, std::size_t n) {
    float32x4_t vb = vdupq_n_f32(b);
    std::size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        float32x4_t a0 = vld1q_f32(a + i +  0);
        float32x4_t a1 = vld1q_f32(a + i +  4);
        float32x4_t a2 = vld1q_f32(a + i +  8);
        float32x4_t a3 = vld1q_f32(a + i + 12);
        vst1q_f32(dst + i +  0, vaddq_f32(a0, vb));
        vst1q_f32(dst + i +  4, vaddq_f32(a1, vb));
        vst1q_f32(dst + i +  8, vaddq_f32(a2, vb));
        vst1q_f32(dst + i + 12, vaddq_f32(a3, vb));
    }
    for (; i + 4 <= n; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        vst1q_f32(dst + i, vaddq_f32(va, vb));
    }
    for (; i < n; ++i) dst[i] = a[i] + b;
}

inline void sub_f32_neon_scalar_rhs(float* dst, const float* a, float b, std::size_t n) {
    float32x4_t vb = vdupq_n_f32(b);
    std::size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        float32x4_t a0 = vld1q_f32(a + i +  0);
        float32x4_t a1 = vld1q_f32(a + i +  4);
        float32x4_t a2 = vld1q_f32(a + i +  8);
        float32x4_t a3 = vld1q_f32(a + i + 12);
        vst1q_f32(dst + i +  0, vsubq_f32(a0, vb));
        vst1q_f32(dst + i +  4, vsubq_f32(a1, vb));
        vst1q_f32(dst + i +  8, vsubq_f32(a2, vb));
        vst1q_f32(dst + i + 12, vsubq_f32(a3, vb));
    }
    for (; i + 4 <= n; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        vst1q_f32(dst + i, vsubq_f32(va, vb));
    }
    for (; i < n; ++i) dst[i] = a[i] - b;
}

inline void mul_f32_neon_scalar_rhs(float* dst, const float* a, float b, std::size_t n) {
    float32x4_t vb = vdupq_n_f32(b);
    std::size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        float32x4_t a0 = vld1q_f32(a + i +  0);
        float32x4_t a1 = vld1q_f32(a + i +  4);
        float32x4_t a2 = vld1q_f32(a + i +  8);
        float32x4_t a3 = vld1q_f32(a + i + 12);
        vst1q_f32(dst + i +  0, vmulq_f32(a0, vb));
        vst1q_f32(dst + i +  4, vmulq_f32(a1, vb));
        vst1q_f32(dst + i +  8, vmulq_f32(a2, vb));
        vst1q_f32(dst + i + 12, vmulq_f32(a3, vb));
    }
    for (; i + 4 <= n; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        vst1q_f32(dst + i, vmulq_f32(va, vb));
    }
    for (; i < n; ++i) dst[i] = a[i] * b;
}

inline void div_f32_neon_scalar_rhs(float* dst, const float* a, float b, std::size_t n) {
    float32x4_t vb = vdupq_n_f32(b);
    std::size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        float32x4_t a0 = vld1q_f32(a + i +  0);
        float32x4_t a1 = vld1q_f32(a + i +  4);
        float32x4_t a2 = vld1q_f32(a + i +  8);
        float32x4_t a3 = vld1q_f32(a + i + 12);
        vst1q_f32(dst + i +  0, vdivq_f32(a0, vb));
        vst1q_f32(dst + i +  4, vdivq_f32(a1, vb));
        vst1q_f32(dst + i +  8, vdivq_f32(a2, vb));
        vst1q_f32(dst + i + 12, vdivq_f32(a3, vb));
    }
    for (; i + 4 <= n; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        vst1q_f32(dst + i, vdivq_f32(va, vb));
    }
    for (; i < n; ++i) dst[i] = a[i] / b;
}

#else // --------- Fallback (non-NEON builds) ---------

inline void add_f32_neon(float* dst, const float* a, const float* b, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) dst[i] = a[i] + b[i];
}
inline void sub_f32_neon(float* dst, const float* a, const float* b, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) dst[i] = a[i] - b[i];
}
inline void mul_f32_neon(float* dst, const float* a, const float* b, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) dst[i] = a[i] * b[i];
}
inline void div_f32_neon(float* dst, const float* a, const float* b, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) dst[i] = a[i] / b[i];
}

inline void add_f32_neon_scalar_rhs(float* dst, const float* a, float b, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) dst[i] = a[i] + b;
}
inline void sub_f32_neon_scalar_rhs(float* dst, const float* a, float b, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) dst[i] = a[i] - b;
}
inline void mul_f32_neon_scalar_rhs(float* dst, const float* a, float b, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) dst[i] = a[i] * b;
}
inline void div_f32_neon_scalar_rhs(float* dst, const float* a, float b, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) dst[i] = a[i] / b;
}
#endif

} // namespace simd

#endif // FUSION_VEC128_NEON_H
