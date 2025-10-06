#ifndef CHECKS_H
#define CHECKS_H
#pragma once
#include <sstream>
#include <stdexcept>

#define FUSION_CHECK(cond, msg)                                                \
  do {                                                                         \
    if (!(cond)) {                                                             \
      std::ostringstream _oss;                                                 \
      _oss << "FUSION_CHECK failed at " << __FILE__ << ":" << __LINE__ << ": " \
           << msg;                                                             \
      throw std::runtime_error(_oss.str());                                    \
    }                                                                          \
  } while (0)

#define FUSION_BOUNDS_CHECK(idx, size)                                         \
  do {                                                                         \
    long long _i = static_cast<long long>(idx);                                \
    long long _n = static_cast<long long>(size);                               \
    if (_i < 0 || _i >= _n) {                                                  \
      std::ostringstream _oss;                                                 \
      _oss << "Index " << _i << " out of bounds for size " << _n << " at "     \
           << __FILE__ << ":" << __LINE__;                                     \
      throw std::out_of_range(_oss.str());                                     \
    }                                                                          \
  } while (0)

#endif // CHECKS_H
