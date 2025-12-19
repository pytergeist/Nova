#ifndef FUSION_CPU_SIMD_TRAITS_HPP
#define FUSION_CPU_SIMD_TRAITS_HPP

#include <cstddef>

#include "SimdTags.hpp"
#include "simd/VecNeon128.hpp"

/* TODO: evaluate the use of neon_scalar for non-commutative operations */

template <class Tag, typename T> struct simd_traits {
   static constexpr bool available = false;
};

// ---------- Addition ----------
template <typename T> struct simd_traits<AddSIMD, T> {
   static constexpr bool available = true;

   // a, b, out are contiguous spans of length n.
   // a_scalar/b_scalar indicate broadcasted scalars.
   static void execute_contiguous(const T *a, const T *b, T *out, std::size_t n,
                                  bool a_scalar, bool b_scalar) {
      if (b_scalar) {
         simd::add_neon_scalar<T>(out, a, *b, n);
      } else if (a_scalar) {
         simd::add_neon_scalar<T>(out, b, *a, n);
      } else {
         simd::add_neon<T>(out, a, b, n);
      }
   }
};

// ---------- Subtraction ----------
template <typename T> struct simd_traits<SubtractSIMD, T> {
   static constexpr bool available = true;

   static void execute_contiguous(const T *a, const T *b, T *out, std::size_t n,
                                  bool a_scalar, bool b_scalar) {
      if (b_scalar) {
         simd::sub_neon_scalar<T>(out, a, *b, n);
      } else if (a_scalar) {
         simd::sub_neon_scalar<T>(out, b, *a, n);
      } else {
         simd::sub_neon<T>(out, a, b, n);
      }
   }
};

// ---------- Division ----------
template <typename T> struct simd_traits<DivideSIMD, T> {
   static constexpr bool available = true;

   static void execute_contiguous(const T *a, const T *b, T *out, std::size_t n,
                                  bool a_scalar, bool b_scalar) {
      if (b_scalar) {
         simd::div_neon_scalar<T>(out, a, *b, n);
      } else if (a_scalar) {
         simd::div_neon_scalar<T>(out, b, *a, n);
      } else {
         simd::div_neon<T>(out, a, b, n);
      }
   }
};

// ---------- Multiply ----------
template <typename T> struct simd_traits<MultiplySIMD, T> {
   static constexpr bool available = true;

   static void execute_contiguous(const T *a, const T *b, T *out, std::size_t n,
                                  bool a_scalar, bool b_scalar) {
      if (b_scalar) {
         simd::mul_neon_scalar<T>(out, a, *b, n);
      } else if (a_scalar) {
         simd::mul_neon_scalar<T>(out, b, *a, n);
      } else {
         simd::mul_neon<T>(out, a, b, n);
      }
   }
};

// ---------- Maximum ----------
template <typename T> struct simd_traits<MaximumSIMD, T> {
   static constexpr bool available = true;

   static void execute_contiguous(const T *a, const T *b, T *out, std::size_t n,
                                  bool a_scalar, bool b_scalar) {
      if (b_scalar) {
         simd::maximum_neon_scalar<T>(out, a, *b, n);
      } else if (a_scalar) {
         simd::maximum_neon_scalar<T>(out, b, *a, n);
      } else {
         simd::maximum_neon<T>(out, a, b, n);
      }
   }
};

// ---------- Power ----------
template <typename T> struct simd_traits<PowerSIMD, T> {
   static constexpr bool available = true;

   static void execute_contiguous(const T *a, const T *b, T *out, std::size_t n,
                                  bool a_scalar, bool b_scalar) {
      if (b_scalar) {
         simd::pow_neon_scalar<T>(out, a, *b, n);
      } else if (a_scalar) {
         simd::pow_neon_scalar<T>(out, b, *a, n);
      } else {
         simd::pow_neon<T>(out, a, b, n);
      }
   }
};

// ---------- Greater Than Equal Too ----------
template <typename T> struct simd_traits<GreaterThanEqualSIMD, T> {
   static constexpr bool available = true;

   static void execute_contiguous(const T *a, const T *b, T *out, std::size_t n,
                                  bool a_scalar, bool b_scalar) {
      if (b_scalar) {
         simd::greater_than_equal_neon_scalar<T>(out, a, *b, n);
      } else if (a_scalar) {
         simd::greater_than_equal_neon_scalar<T>(out, b, *a, n);
      } else {
         simd::greater_than_equal_neon<T>(out, a, b, n);
      }
   }
};

// ---------- Greater Than ----------
template <typename T> struct simd_traits<GreaterThanSIMD, T> {
   static constexpr bool available = true;

   static void execute_contiguous(const T *a, const T *b, T *out, std::size_t n,
                                  bool a_scalar, bool b_scalar) {
      if (b_scalar) {
         simd::greater_than_neon_scalar<T>(out, a, *b, n);
      } else if (a_scalar) {
         simd::greater_than_neon_scalar<T>(out, b, *a, n);
      } else {
         simd::greater_than_neon<T>(out, a, b, n);
      }
   }
};

// ---------- Exponential ----------
template <typename T> struct simd_traits<ExponentialSIMD, T> {
   static constexpr bool available = true;

   static void execute_contiguous(const T *a, T *out, std::size_t n,
                                  bool a_scalar) {
      simd::exp_neon<T>(out, a, n);
   }
};

// ---------- Natural Log ----------
template <typename T> struct simd_traits<NaturalLogSIMD, T> {
   static constexpr bool available = true;

   static void execute_contiguous(const T *a, T *out, std::size_t n,
                                  bool a_scalar) {
      simd::log_neon<T>(out, a, n);
   }
};

// ---------- Sqrt ----------
template <typename T> struct simd_traits<SqrtSIMD, T> {
   static constexpr bool available = true;

   static void execute_contiguous(const T *a, T *out, std::size_t n,
                                  bool a_scalar) {
      simd::sqrt_neon<T>(out, a, n);
   }
};

// ---------- Global sum ----------
template <typename T> struct simd_traits<GlobalSumSIMD, T> {
   static constexpr bool available = true;

   static T reduce(const T *a, std::size_t n) {
      T acc = T(0.0);
      simd::sum_neon<T>(&acc, a, n);
      return acc;
   }
};
#endif // FUSION_CPU_SIMD_TRAITS_HPP
