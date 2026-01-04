// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2025 Tom Pope
//
// Nova — a high-performance hybrid physics and deep learning tensor engine.

#ifndef FUSION_CPU_VEC_FALLBACK_HPP
#define FUSION_CPU_VEC_FALLBACK_HPP

#include <cmath>
#include <cstddef>
#include <cstdint>

namespace simd {

template <typename T>
inline void
greater_than_equal_contiguous(T *__restrict dst, const T *__restrict a,
                              const T *__restrict b, std::size_t n) {
   for (std::size_t i = 0; i < n; ++i)
      dst[i] = a[i] >= b[i];
}

template <typename T>
inline void greater_than_contiguous(T *__restrict dst, const T *__restrict a,
                                    const T *__restrict b, std::size_t n) {
   for (std::size_t i = 0; i < n; ++i)
      dst[i] = a[i] > b[i];
}

template <typename T>
inline void greater_than_contiguous_scalar(T *__restrict dst,
                                           const T *__restrict a, T b,
                                           std::size_t n) {
   for (std::size_t i = 0; i < n; ++i)
      dst[i] = a[i] > b;
}

template <typename T>
inline void sum_contiguous(T *dst, const T *a, std::size_t n) {
   T acc = 0.0f;
   for (std::size_t i = 0; i < n; ++i)
      acc += a[i];
   *dst = acc;
}

template <typename T>
inline void sqrt_contiguous(T *__restrict dst, const T *__restrict a,
                            std::size_t n) {
   for (std::size_t i = 0; i < n; ++i)
      dst[i] = std::sqrt(a[i]);
}

template <typename T>
inline void log_contiguous(T *__restrict dst, const T *__restrict a,
                           std::size_t n) {
   for (std::size_t i = 0; i < n; ++i)
      dst[i] = std::log(a[i]);
}

template <typename T>
inline void exp_contiguous(T *__restrict dst, const T *__restrict a,
                           std::size_t n) {
   for (std::size_t i = 0; i < n; ++i)
      dst[i] = std::exp(a[i]);
}

template <typename T>
inline void pow_contiguous(T *__restrict__ dst, const T *__restrict__ a,
                           const T *b, std::size_t n) {
   for (std::size_t i = 0; i < n; ++i)
      dst[i] = std::pow(a[i], b[i]);
}

template <typename T>
inline void maximum_contiguous(T *__restrict dst, const T *__restrict a,
                               const T *__restrict b, std::size_t n) {
   for (std::size_t i = 0; i < n; ++i)
      dst[i] = a[i] > b[i] ? a[i] : b[i];
}

template <typename T>
inline void add_contiguous(T *__restrict dst, const T *__restrict a,
                           const T *__restrict b, std::size_t n) {
   for (std::size_t i = 0; i < n; ++i)
      dst[i] = a[i] + b[i];
}

template <typename T>
inline void sub_contiguous(T *__restrict dst, const T *__restrict a,
                           const T *__restrict b, std::size_t n) {
   for (std::size_t i = 0; i < n; ++i)
      dst[i] = a[i] - b[i];
}

template <typename T>
inline void mul_contiguous(T *__restrict dst, const T *__restrict a,
                           const T *__restrict b, std::size_t n) {
   for (std::size_t i = 0; i < n; ++i)
      dst[i] = a[i] * b[i];
}

template <typename T>
inline void div_contiguous(T *__restrict dst, const T *__restrict a,
                           const T *__restrict b, std::size_t n) {
   for (std::size_t i = 0; i < n; ++i)
      dst[i] = a[i] / b[i];
}

template <typename T>
inline void greater_than_equal_contiguous_scalar(T *__restrict dst,
                                                 const T *__restrict a, T b,
                                                 std::size_t n) {
   for (std::size_t i = 0; i < n; ++i)
      dst[i] = a[i] >= b;
}

template <typename T>
inline void pow_contiguous_scalar(T *__restrict dst, const T *__restrict a, T b,
                                  std::size_t n) {
   for (std::size_t i = 0; i < n; ++i)
      dst[i] = std::pow(a[i], b);
}

template <typename T>
inline void add_contiguous_scalar(T *__restrict dst, const T *__restrict a, T b,
                                  std::size_t n) {
   for (std::size_t i = 0; i < n; ++i)
      dst[i] = a[i] + b;
}

template <typename T>
inline void sub_contiguous_scalar(T *__restrict dst, const T *__restrict a, T b,
                                  std::size_t n) {
   for (std::size_t i = 0; i < n; ++i)
      dst[i] = a[i] - b;
}

template <typename T>
inline void mul_contiguous_scalar(T *__restrict dst, const T *__restrict a, T b,
                                  std::size_t n) {
   for (std::size_t i = 0; i < n; ++i)
      dst[i] = a[i] * b;
}

template <typename T>
inline void div_contiguous_scalar(T *__restrict dst, const T *__restrict a, T b,
                                  std::size_t n) {
   for (std::size_t i = 0; i < n; ++i)
      dst[i] = a[i] / b;
}

template <typename T>
inline void maximum_contiguous_scalar(T *__restrict dst, const T *__restrict a,
                                      const T b, std::size_t n) {
   for (std::size_t i = 0; i < n; ++i)
      dst[i] = a[i] > b ? a[i] : b;
}

} // namespace simd

#endif // FUSION_CPU_VEC_FALLBACK_HPP
