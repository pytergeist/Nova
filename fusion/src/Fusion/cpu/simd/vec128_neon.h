#ifndef VEC128_NEON_H
#define VEC128_NEON_H
#pragma once

#include <arm_neon.h>
#include <cstddef>

namespace simd {

    // Primary template (no definition)
    template <typename T>
    void vec128_addition_neon(T* dst,
                              T const* a,
                              T const* b,
                              std::size_t na,
                              std::size_t nb);

    // Inline float‐specialization
    template <>
    inline void vec128_addition_neon<float>(float*       dst,
                                            float const* a,
                                            float const* b,
                                            std::size_t  na,
                                            std::size_t  nb) {
        const bool a_is_scalar = (na == 1);
        const bool b_is_scalar = (nb == 1);
        const std::size_t N   = a_is_scalar ? nb : na;

        float32x4_t a_dup, b_dup;
        if (a_is_scalar) a_dup = vdupq_n_f32(a[0]);
        if (b_is_scalar) b_dup = vdupq_n_f32(b[0]);

        std::size_t i = 0;
        const std::size_t vec_end = (N / 4) * 4;

        // Vectorized loop
        for (; i < vec_end; i += 4) {
            float32x4_t va = a_is_scalar ? a_dup : vld1q_f32(a + i);
            float32x4_t vb = b_is_scalar ? b_dup : vld1q_f32(b + i);
            vst1q_f32(dst + i, vaddq_f32(va, vb));
        }
        // Remainder
        for (; i < N; ++i) {
            float fa = a_is_scalar ? a[0] : a[i];
            float fb = b_is_scalar ? b[0] : b[i];
            dst[i]  = fa + fb;
        }
    }

} // namespace simd

#endif // VEC128_NEON_H
