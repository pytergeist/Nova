#ifndef SIMD_TRAITS_H
#define SIMD_TRAITS_H

#include <type_traits>
#include "simd/vec128_neon.h"
#include "simd_tags.h"

template <class Tag, typename T>
struct simd_traits {
  static constexpr bool available = false;
};


template <>
struct simd_traits<AddSIMD, float> {
  static constexpr bool available = true;

  template <class FFuncT>
  static void execute(FFuncT const &f, float *out) {
    auto const &lhs_t = f.template operand<0>();
    auto const &rhs_t = f.template operand<1>();

    const float *lhs = lhs_t.raw_data().data();
    const float *rhs = rhs_t.raw_data().data();
    size_t na = lhs_t.flat_size();
    size_t nb = rhs_t.flat_size();

    simd::vec128_addition_neon(out, lhs, rhs, na, nb);
  }
};


template <>
struct simd_traits<SubtractSIMD, float> {
  static constexpr bool available = true;

  template <class FFuncT>
  static void execute(FFuncT const &f, float *out) {
    auto const &lhs_t = f.template operand<0>();
    auto const &rhs_t = f.template operand<1>();

    const float *lhs = lhs_t.raw_data().data();
    const float *rhs = rhs_t.raw_data().data();
    size_t na = lhs_t.flat_size();
    size_t nb = rhs_t.flat_size();

    simd::vec128_addition_neon(out, lhs, rhs, na, nb);
  }
};




#endif // SIMD_TRAITS_H
