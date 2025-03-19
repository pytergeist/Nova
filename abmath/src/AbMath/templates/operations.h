#ifndef ABMATH_OPERATIONS_H
#define ABMATH_OPERATIONS_H

#include "vector.h"

namespace abmath {

template <typename A, typename B>
auto add(const A &a, const B &b) -> decltype(a + b) {
  return a + b;
}
} // namespace abmath

#endif
