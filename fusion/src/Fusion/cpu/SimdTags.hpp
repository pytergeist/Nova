#ifndef FUSION_CPU_SIMD_TAGS_HPP
#define FUSION_CPU_SIMD_TAGS_HPP

#include <cmath>
#include <type_traits>

struct AddSIMD {
   template <typename U> constexpr U operator()(U a, U b) const noexcept {
      return a + b;
   }
};

struct SubtractSIMD {
   template <typename U> constexpr U operator()(U a, U b) const noexcept {
      return a - b;
   }
};

struct DivideSIMD {
   template <typename U> constexpr U operator()(U a, U b) const noexcept {
      return a / b;
   }
};

struct MultiplySIMD {
   template <typename U> constexpr U operator()(U a, U b) const noexcept {
      return a * b;
   }
};

struct MaximumSIMD {
   template <typename U> constexpr U operator()(U a, U b) const noexcept {
      return a > b ? a : b;
   }
};

struct PowerSIMD {
   template <typename U> constexpr U operator()(U a, U b) const noexcept {
      return std::pow(a, b);
   }
};

struct GreaterThanEqualSIMD {
   template <typename U> constexpr U operator()(U a, U b) const noexcept {
      return a >= b;
   }
};

struct GreaterThanSIMD {
   template <typename U> constexpr U operator()(U a, U b) const noexcept {
      return a > b;
   }
};

struct ExponentialSIMD {
   template <typename U> constexpr U operator()(U a) const noexcept {
      return std::exp(a);
   }
};

struct NaturalLogSIMD {
   template <typename U> constexpr U operator()(U a) const noexcept {
      return std::log(a);
   }
};

struct SqrtSIMD {
   template <typename U> constexpr U operator()(U a) const noexcept {
      return std::sqrt(a);
   }
};

struct SumSIMD {
   template <typename U> constexpr U operator()(U a) const noexcept {
      return a;
   }
};


#endif // FUSION_CPU_SIMD_TAGS_HPP
