#ifndef SIMD_TRAITS_H
#define SIMD_TRAITS_H

#include <cstddef>
#include "simd/vec128_neon.h"
#include "simd_tags.h"

template<class Tag, typename T>
struct simd_traits {
    static constexpr bool available = false;
};

// ---------- Addition ----------
template<>
struct simd_traits<AddSIMD, float> {
    static constexpr bool available = true;

    // a, b, out are contiguous spans of length n.
    // a_scalar/b_scalar indicate broadcasted scalars.
    static void execute_contiguous(const float *a, const float *b, float *out,
                                   std::size_t n, bool a_scalar, bool b_scalar) {
        if (b_scalar) {
            simd::add_f32_neon_scalar_rhs(out, a, *b, n);
        } else if (a_scalar) {
            simd::add_f32_neon_scalar_rhs(out, b, *a, n);
        } else {
            simd::add_f32_neon(out, a, b, n);
        }
    }
};

// ---------- Subtraction ----------
template<>
struct simd_traits<SubtractSIMD, float> {
    static constexpr bool available = true;

    static void execute_contiguous(const float *a, const float *b, float *out,
                                   std::size_t n, bool a_scalar, bool b_scalar) {
        if (b_scalar) {
            simd::sub_f32_neon_scalar_rhs(out, a, *b, n);
        } else if (a_scalar) {
            simd::sub_f32_neon_scalar_rhs(out, b, *a, n);
        } else {
            simd::sub_f32_neon(out, a, b, n);
        }
    }
};


// ---------- Division ----------
template<>
struct simd_traits<DivideSIMD, float> {
    static constexpr bool available = true;

    static void execute_contiguous(const float *a, const float *b, float *out,
                                   std::size_t n, bool a_scalar, bool b_scalar) {
        if (b_scalar) {
            simd::div_f32_neon_scalar_rhs(out, a, *b, n);
        } else if (a_scalar) {
            simd::div_f32_neon_scalar_rhs(out, b, *a, n);
        } else {
            simd::div_f32_neon(out, a, b, n);
        }
    }
};


// ---------- Multiply ----------
template<>
struct simd_traits<MultiplySIMD, float> {
    static constexpr bool available = true;

    static void execute_contiguous(const float *a, const float *b, float *out,
                                   std::size_t n, bool a_scalar, bool b_scalar) {
        if (b_scalar) {
            simd::mul_f32_neon_scalar_rhs(out, a, *b, n);
        } else if (a_scalar) {
            simd::mul_f32_neon_scalar_rhs(out, b, *a, n);
        } else {
            simd::mul_f32_neon(out, a, b, n);
        }
    }
};

// ---------- Maximum ----------
template<>
struct simd_traits<MaximumSIMD, float> {
    static constexpr bool available = true;

    static void execute_contiguous(const float *a, const float *b, float *out,
                                   std::size_t n, bool a_scalar, bool b_scalar) {
        if (b_scalar) {
            simd::maximum_f32_neon_scalar_rhs(out, a, *b, n);
        } else if (a_scalar) {
            simd::maximum_f32_neon_scalar_rhs(out, b, *a, n);
        } else {
            simd::maximum_f32_neon(out, a, b, n);
        }
    }
};

#endif // SIMD_TRAITS_H
