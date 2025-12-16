#ifndef SIMD_TRAITS_HPP
#define SIMD_TRAITS_HPP

#include <cstddef>

#include "SimdTags.hpp"
#include "simd/Vec128Neon.hpp"

template <class Tag, typename T> struct simd_traits {
   static constexpr bool available = false;
};

// ---------- Addition ----------
template <> struct simd_traits<AddSIMD, float> {
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
template <> struct simd_traits<SubtractSIMD, float> {
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
template <> struct simd_traits<DivideSIMD, float> {
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
template <> struct simd_traits<MultiplySIMD, float> {
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
template <> struct simd_traits<MaximumSIMD, float> {
   static constexpr bool available = true;

   static void execute_contiguous(const float *a, const float *b, float *out,
                                  std::size_t n, bool a_scalar, bool b_scalar) {
      if (b_scalar) {
         simd::maximum_f32_neon_scalar_rhs(out, a, *b, n);
      } else if (a_scalar) {
         simd::maximum_f32_neon_scalar_rhs(out, b, *a,
                                           n); // TODO: rename from rhs
      } else {
         simd::maximum_f32_neon(out, a, b, n);
      }
   }
};

// ---------- Power ----------
template <> struct simd_traits<PowerSIMD, float> {
   static constexpr bool available = true;

   static void execute_contiguous(const float *a, const float *b, float *out,
                                  std::size_t n, bool a_scalar, bool b_scalar) {
      if (b_scalar) {
         simd::pow_f32_neon_scalar_rhs(out, a, *b, n);
      } else if (a_scalar) {
         simd::pow_f32_neon_scalar_rhs(out, b, *a, n);
      } else {
         simd::pow_f32_neon(out, a, b, n);
      }
   }
};

// ---------- Greater Than Equal Too ----------
template <> struct simd_traits<GreaterThanEqualSIMD, float> {
   static constexpr bool available = true;

   static void execute_contiguous(const float *a, const float *b, float *out,
                                  std::size_t n, bool a_scalar, bool b_scalar) {
      if (b_scalar) {
         simd::greater_than_equal_f32_neon_scalar(out, a, *b, n);
      } else if (a_scalar) {
         simd::greater_than_equal_f32_neon_scalar(out, b, *a, n);
      } else {
         simd::greater_than_equal_f32_neon(out, a, b, n);
      }
   }
};

// ---------- Greater Than ----------
template <> struct simd_traits<GreaterThanSIMD, float> {
   static constexpr bool available = true;

   static void execute_contiguous(const float *a, const float *b, float *out,
                                  std::size_t n, bool a_scalar, bool b_scalar) {
      if (b_scalar) {
         simd::greater_than_f32_neon_scalar(out, a, *b, n);
      } else if (a_scalar) {
         simd::greater_than_f32_neon_scalar(out, b, *a, n);
      } else {
         simd::greater_than_f32_neon(out, a, b, n);
      }
   }
};

// ---------- Exponential ----------
template <> struct simd_traits<ExponentialSIMD, float> {
   static constexpr bool available = true;

   static void execute_contiguous(const float *a, float *out, std::size_t n,
                                  bool a_scalar) {
      simd::exp_f32_neon(out, a, n);
   }
};

// ---------- Natural Log ----------
template <> struct simd_traits<NaturalLogSIMD, float> {
   static constexpr bool available = true;

   static void execute_contiguous(const float *a, float *out, std::size_t n,
                                  bool a_scalar) {
      simd::log_f32_neon(out, a, n);
   }
};

// ---------- Sqrt ----------
template <> struct simd_traits<SqrtSIMD, float> {
   static constexpr bool available = true;

   static void execute_contiguous(const float *a, float *out, std::size_t n,
                                  bool a_scalar) {
      simd::sqrt_f32_neon(out, a, n);
   }
};

// ---------- Global sum ----------
template <> struct simd_traits<GlobalSumSIMD, float> {
   static constexpr bool available = true;

   static float reduce(const float *a, std::size_t n) {
      float acc = 0.0f;
      simd::sum_f32_neon(&acc, a, n);
      return acc;
   }
};
#endif // SIMD_TRAITS_HPP
