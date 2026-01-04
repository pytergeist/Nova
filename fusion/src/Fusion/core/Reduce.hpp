// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2025 Tom Pope
//
// Nova — a high-performance hybrid physics and deep learning tensor engine.

#ifndef CORE_REDUCE_HPP
#define CORE_REDUCE_HPP

#include "Fusion/cpu/SimdTraits.hpp"

namespace reduce {
template <typename T, class Tag> T reduce_tag(const T *x, std::size_t n) {
   if constexpr (simd_traits<Tag, T>::available) {
      return simd_traits<Tag, T>::reduce(x, n);
   } else {
      T acc = T{0};
      for (std::size_t i = 0; i < n; ++i)
         acc += x[i];
      return acc;
   }
}
} // namespace reduce

#endif // CORE_REDUCE_HPP
