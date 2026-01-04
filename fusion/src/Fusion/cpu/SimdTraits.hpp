#ifndef FUSION_CPU_SIMD_TRAITS_HPP
#define FUSION_CPU_SIMD_TRAITS_HPP

#include <cstddef>

#include "SimdTags.hpp"

#if defined(FUSION_ENABLE_NEON) && defined(__ARM_NEON)
#include "simd/VecNeon128.hpp"
#else
#include "simd/VecFallback.hpp"
#endif

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
         simd::add_contiguous_scalar<T>(out, a, *b, n);
      } else if (a_scalar) {
         simd::add_contiguous_scalar<T>(out, b, *a, n);
      } else {
         simd::add_contiguous<T>(out, a, b, n);
      }
   }
};

// ---------- Subtraction ----------
template <typename T> struct simd_traits<SubtractSIMD, T> {
   static constexpr bool available = true;

   static void execute_contiguous(const T *a, const T *b, T *out, std::size_t n,
                                  bool a_scalar, bool b_scalar) {
      if (b_scalar) {
         simd::sub_contiguous_scalar<T>(out, a, *b, n);
      } else if (a_scalar) {
         simd::sub_contiguous_scalar<T>(out, b, *a, n);
      } else {
         simd::sub_contiguous<T>(out, a, b, n);
      }
   }
};

// ---------- Division ----------
template <typename T> struct simd_traits<DivideSIMD, T> {
   static constexpr bool available = true;

   static void execute_contiguous(const T *a, const T *b, T *out, std::size_t n,
                                  bool a_scalar, bool b_scalar) {
      if (b_scalar) {
         simd::div_contiguous_scalar<T>(out, a, *b, n);
      } else if (a_scalar) {
         simd::div_contiguous_scalar<T>(out, b, *a, n);
      } else {
         simd::div_contiguous<T>(out, a, b, n);
      }
   }
};

// ---------- Multiply ----------
template <typename T> struct simd_traits<MultiplySIMD, T> {
   static constexpr bool available = true;

   static void execute_contiguous(const T *a, const T *b, T *out, std::size_t n,
                                  bool a_scalar, bool b_scalar) {
      if (b_scalar) {
         simd::mul_contiguous_scalar<T>(out, a, *b, n);
      } else if (a_scalar) {
         simd::mul_contiguous_scalar<T>(out, b, *a, n);
      } else {
         simd::mul_contiguous<T>(out, a, b, n);
      }
   }
};

// ---------- Maximum ----------
template <typename T> struct simd_traits<MaximumSIMD, T> {
   static constexpr bool available = true;

   static void execute_contiguous(const T *a, const T *b, T *out, std::size_t n,
                                  bool a_scalar, bool b_scalar) {
      if (b_scalar) {
         simd::maximum_contiguous_scalar<T>(out, a, *b, n);
      } else if (a_scalar) {
         simd::maximum_contiguous_scalar<T>(out, b, *a, n);
      } else {
         simd::maximum_contiguous<T>(out, a, b, n);
      }
   }
};

// ---------- Power ----------
template <typename T> struct simd_traits<PowerSIMD, T> {
   static constexpr bool available = true;

   static void execute_contiguous(const T *a, const T *b, T *out, std::size_t n,
                                  bool a_scalar, bool b_scalar) {
      if (b_scalar) {
         simd::pow_contiguous_scalar<T>(out, a, *b, n);
      } else if (a_scalar) {
         simd::pow_contiguous_scalar<T>(out, b, *a, n);
      } else {
         simd::pow_contiguous<T>(out, a, b, n);
      }
   }
};

// ---------- Greater Than Equal Too ----------
template <typename T> struct simd_traits<GreaterThanEqualSIMD, T> {
   static constexpr bool available = true;

   static void execute_contiguous(const T *a, const T *b, T *out, std::size_t n,
                                  bool a_scalar, bool b_scalar) {
      if (b_scalar) {
         simd::greater_than_equal_contiguous_scalar<T>(out, a, *b, n);
      } else if (a_scalar) {
         simd::greater_than_equal_contiguous_scalar<T>(out, b, *a, n);
      } else {
         simd::greater_than_equal_contiguous<T>(out, a, b, n);
      }
   }
};

// ---------- Greater Than ----------
template <typename T> struct simd_traits<GreaterThanSIMD, T> {
   static constexpr bool available = true;

   static void execute_contiguous(const T *a, const T *b, T *out, std::size_t n,
                                  bool a_scalar, bool b_scalar) {
      if (b_scalar) {
         simd::greater_than_contiguous_scalar<T>(out, a, *b, n);
      } else if (a_scalar) {
         simd::greater_than_contiguous_scalar<T>(out, b, *a, n);
      } else {
         simd::greater_than_contiguous<T>(out, a, b, n);
      }
   }
};

// ---------- Exponential ----------
template <typename T> struct simd_traits<ExponentialSIMD, T> {
   static constexpr bool available = true;

   static void execute_contiguous(const T *a, T *out, std::size_t n,
                                  bool a_scalar) {
      simd::exp_contiguous<T>(out, a, n);
   }
};

// ---------- Natural Log ----------
template <typename T> struct simd_traits<NaturalLogSIMD, T> {
   static constexpr bool available = true;

   static void execute_contiguous(const T *a, T *out, std::size_t n,
                                  bool a_scalar) {
      simd::log_contiguous<T>(out, a, n);
   }
};

// ---------- Sqrt ----------
template <typename T> struct simd_traits<SqrtSIMD, T> {
   static constexpr bool available = true;

   static void execute_contiguous(const T *a, T *out, std::size_t n,
                                  bool a_scalar) {
      simd::sqrt_contiguous<T>(out, a, n);
   }
};

// ---------- Sum ----------
template <typename T> struct simd_traits<SumSIMD, T> {
   static constexpr bool available = true;

   static T reduce_contiguous(const T *a, std::size_t n) {
      T acc = T(0.0);
      simd::sum_contiguous<T>(&acc, a, n);
      return acc;
   }
};

#endif // FUSION_CPU_SIMD_TRAITS_HPP
