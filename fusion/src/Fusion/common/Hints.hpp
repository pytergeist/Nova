#ifndef HINTS_HPP
#define HINTS_HPP
#pragma once

#include <cstddef>

namespace fusion::detail {

template <typename T, std::size_t Align>
inline const T *const_assume_aligned(const T *ptr) {
#if defined(__clang__) || defined(__GNUC__)
   ptr = static_cast<const T *>(__builtin_assume_aligned(ptr, Align));
   return ptr;
#else
   return ptr;
#endif
};

template <typename T, std::size_t Align> inline T *assume_aligned(T *ptr) {
#if defined(__clang__) || defined(__GNUC__)
   ptr = static_cast<T *>(__builtin_assume_aligned(ptr, Align));
   return ptr;
#else
   return ptr;
#endif
};

} // namespace fusion::detail

#define FUSION_ASSUME_ALIGNED(type, ptr, align)                                \
   (ptr = ::fusion::detail::assume_aligned<type, align>(ptr))

#define FUSION_CONST_ASSUME_ALIGNED(type, ptr, align)                          \
   (ptr = ::fusion::detail::const_assume_aligned<type, align>(ptr))

#endif // HINTS_HPP
