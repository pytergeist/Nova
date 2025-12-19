#ifndef FUSION_CPU_VEC_FALLBACK_HPP
#define FUSION_CPU_VEC_FALLBACK_HPP

#include <cmath>
#include <cstddef>
#include <cstdint>

template <typename T>
inline void greater_than_equal_neon(T *__restrict dst, const T *__restrict a,
                                    const T *__restrict b, std::size_t n) {
   for (std::size_t i = 0; i < n; ++i)
      dst[i] = a[i] >= b[i];
}

template <typename T>
inline void greater_than_neon(T *__restrict dst, const T *__restrict a,
                              const T *__restrict b, std::size_t n) {
   for (std::size_t i = 0; i < n; ++i)
      dst[i] = a[i] > b[i];
}

template <typename T>
inline void greater_than_neon_scalar(T *__restrict dst, const T *__restrict a,
                                     T b, std::size_t n) {
   for (std::size_t i = 0; i < n; ++i)
      dst[i] = a[i] > b;
}

template <typename T> inline void sum_neon(T *dst, const T *a, std::size_t n) {
   T acc = 0.0f;
   for (std::size_t i = 0; i < n; ++i)
      acc += a[i];
   *dst = acc;
}

template <typename T>
inline void sqrt_neon(T *__restrict dst, const T *__restrict a, std::size_t n) {
   for (std::size_t i = 0; i < n; ++i)
      dst[i] = std::sqrt(a[i]);
}

template <typename T>
inline void log_neon(T *__restrict dst, const T *__restrict a, std::size_t n) {
   for (std::size_t i = 0; i < n; ++i)
      dst[i] = std::log(a[i]);
}

template <typename T>
inline void exp_neon(T *__restrict dst, const T *__restrict a, std::size_t n) {
   for (std::size_t i = 0; i < n; ++i)
      dst[i] = std::exp(a[i]);
}

template <typename T>
inline void pow_neon(T *__restrict__ dst, const T *__restrict__ a, const T *b,
                     std::size_t n) {
   for (std::size_t i = 0; i < n; ++i)
      dst[i] = std::pow(a[i], b[i]);
}

template <typename T>
inline void maximum_neon(T *__restrict dst, const T *__restrict a,
                         const T *__restrict b, std::size_t n) {
   for (std::size_t i = 0; i < n; ++i)
      dst[i] = a[i] > b[i] ? a[i] : b[i];
}

template <typename T>
inline void add_neon(T *__restrict dst, const T *__restrict a,
                     const T *__restrict b, std::size_t n) {
   for (std::size_t i = 0; i < n; ++i)
      dst[i] = a[i] + b[i];
}

template <typename T>
inline void sub_neon(T *__restrict dst, const T *__restrict a,
                     const T *__restrict b, std::size_t n) {
   for (std::size_t i = 0; i < n; ++i)
      dst[i] = a[i] - b[i];
}

template <typename T>
inline void mul_neon(T *__restrict dst, const T *__restrict a,
                     const T *__restrict b, std::size_t n) {
   for (std::size_t i = 0; i < n; ++i)
      dst[i] = a[i] * b[i];
}

template <typename T>
inline void div_neon(T *__restrict dst, const T *__restrict a,
                     const T *__restrict b, std::size_t n) {
   for (std::size_t i = 0; i < n; ++i)
      dst[i] = a[i] / b[i];
}

template <typename T>
inline void greater_than_equal_neon_scalar(T *__restrict dst,
                                           const T *__restrict a, T b,
                                           std::size_t n) {
   for (std::size_t i = 0; i < n; ++i)
      dst[i] = a[i] >= b;
}

template <typename T>
inline void pow_neon_scalar(T *__restrict dst, const T *__restrict a, T b,
                            std::size_t n) {
   for (std::size_t i = 0; i < n; ++i)
      dst[i] = std::pow(a[i], b);
}

template <typename T>
inline void add_neon_scalar(T *__restrict dst, const T *__restrict a, T b,
                            std::size_t n) {
   for (std::size_t i = 0; i < n; ++i)
      dst[i] = a[i] + b;
}

template <typename T>
inline void sub_neon_scalar(T *__restrict dst, const T *__restrict a, T b,
                            std::size_t n) {
   for (std::size_t i = 0; i < n; ++i)
      dst[i] = a[i] - b;
}

template <typename T>
inline void mul_neon_scalar(T *__restrict dst, const T *__restrict a, T b,
                            std::size_t n) {
   for (std::size_t i = 0; i < n; ++i)
      dst[i] = a[i] * b;
}

template <typename T>
inline void div_neon_scalar(T *__restrict dst, const T *__restrict a, T b,
                            std::size_t n) {
   for (std::size_t i = 0; i < n; ++i)
      dst[i] = a[i] / b;
}

template <typename T>
inline void maximum_neon_scalar(T *__restrict dst, const T *__restrict a,
                                const T b, std::size_t n) {
   for (std::size_t i = 0; i < n; ++i)
      dst[i] = a[i] > b ? a[i] : b;
}

#endif // FUSION_CPU_VEC_FALLBACK_HPP
