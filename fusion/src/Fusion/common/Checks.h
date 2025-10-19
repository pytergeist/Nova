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

#define FUSION_ALLOW_SCALAR_BINARY(t1, t2)                               \
  do {                                                                         \
    const auto &_t1 = (t1);                                                    \
    const auto &_t2 = (t2);                                                    \
    const auto _sz1 = _t1.size();                                              \
    const auto _sz2 = _t2.size();                                              \
    const bool _is_scalar1 = (_t1.ndims() == 0) || (_sz1 == 1);                 \
    const bool _is_scalar2 = (_t2.ndims() == 0) || (_sz2 == 1);                 \
    if (!(_is_scalar1 || _is_scalar2 || (_sz1 == _sz2))) {                     \
      std::ostringstream _oss;                                                 \
      _oss << "Elementwise op requires scalars or equal element counts; got "  \
           << "size(" << _sz1 << ") vs size(" << _sz2 << ") at " << __FILE__   \
           << ":" << __LINE__;                                                 \
      throw std::runtime_error(_oss.str());                                    \
    }                                                                          \
  } while (0)

#endif // CHECKS_H
