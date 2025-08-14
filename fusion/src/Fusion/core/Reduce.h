#ifndef REDUCE_H
#define REDUCE_H

#pragma once
#include "../cpu/SimdTraits.h"

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

#endif // REDUCE_H
