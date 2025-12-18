#ifndef VEC_FALLBACK_HPP
#define VEC_FALLBACK_HPP

inline void greater_than_equal_f32_neon(float *__restrict dst,
                                        const float *__restrict a,
                                        const float *__restrict b,
                                        std::size_t n) {
   for (std::size_t i = 0; i < n; ++i)
      dst[i] = a[i] >= b[i];
}

inline void greater_than_f32_neon(float *__restrict dst,
                                  const float *__restrict a,
                                  const float *__restrict b, std::size_t n) {
   for (std::size_t i = 0; i < n; ++i)
      dst[i] = a[i] > b[i];
}

inline void greater_than_f32_neon_scalar(float *__restrict dst,
                                         const float *__restrict a, float b,
                                         std::size_t n) {
   for (std::size_t i = 0; i < n; ++i)
      dst[i] = a[i] > b;
}

inline void sum_f32_neon(float *dst, const float *a, std::size_t n) {
   float acc = 0.0f;
   for (std::size_t i = 0; i < n; ++i)
      acc += a[i];
   *dst = acc;
}
inline void sqrt_f32_neon(float *__restrict dst, const float *__restrict a,
                          std::size_t n) {
   for (std::size_t i = 0; i < n; ++i)
      dst[i] = std::sqrt(a[i]);
}

inline void log_f32_neon(float *__restrict dst, const float *__restrict a,
                         std::size_t n) {
   for (std::size_t i = 0; i < n; ++i)
      dst[i] = std::log(a[i]);
}

inline void exp_f32_neon(float *__restrict dst, const float *__restrict a,
                         std::size_t n) {
   for (std::size_t i = 0; i < n; ++i)
      dst[i] = std::exp(a[i]);
}

inline void pow_f32_neon(float *__restrict__ dst, const float *__restrict__ a,
                         const float *b, std::size_t n) {
   for (std::size_t i = 0; i < n; ++i)
      dst[i] = std::pow(a[i], b[i]);
}

inline void maximum_f32_neon(float *__restrict dst, const float *__restrict a,
                             const float *__restrict b, std::size_t n) {
   for (std::size_t i = 0; i < n; ++i)
      dst[i] = a[i] > b[i] ? a[i] : b[i];
}

inline void add_f32_neon(float *__restrict dst, const float *__restrict a,
                         const float *__restrict b, std::size_t n) {
   for (std::size_t i = 0; i < n; ++i)
      dst[i] = a[i] + b[i];
}

inline void sub_f32_neon(float *__restrict dst, const float *__restrict a,
                         const float *__restrict b, std::size_t n) {
   for (std::size_t i = 0; i < n; ++i)
      dst[i] = a[i] - b[i];
}

inline void mul_f32_neon(float *__restrict dst, const float *__restrict a,
                         const float *__restrict b, std::size_t n) {
   for (std::size_t i = 0; i < n; ++i)
      dst[i] = a[i] * b[i];
}

inline void div_f32_neon(float *__restrict dst, const float *__restrict a,
                         const float *__restrict b, std::size_t n) {
   for (std::size_t i = 0; i < n; ++i)
      dst[i] = a[i] / b[i];
}

inline void greater_than_equal_f32_neon_scalar(float *__restrict dst,
                                               const float *__restrict a,
                                               float b, std::size_t n) {
   for (std::size_t i = 0; i < n; ++i)
      dst[i] = a[i] >= b;
}

inline void pow_f32_neon_scalar(float *__restrict dst,
                                const float *__restrict a, float b,
                                std::size_t n) {
   for (std::size_t i = 0; i < n; ++i)
      dst[i] = std::pow(a[i], b);
}

inline void add_f32_neon_scalar(float *__restrict dst,
                                const float *__restrict a, float b,
                                std::size_t n) {
   for (std::size_t i = 0; i < n; ++i)
      dst[i] = a[i] + b;
}

inline void sub_f32_neon_scalar(float *__restrict dst,
                                const float *__restrict a, float b,
                                std::size_t n) {
   for (std::size_t i = 0; i < n; ++i)
      dst[i] = a[i] - b;
}

inline void mul_f32_neon_scalar(float *__restrict dst,
                                const float *__restrict a, float b,
                                std::size_t n) {
   for (std::size_t i = 0; i < n; ++i)
      dst[i] = a[i] * b;
}

inline void div_f32_neon_scalar(float *__restrict dst,
                                const float *__restrict a, float b,
                                std::size_t n) {
   for (std::size_t i = 0; i < n; ++i)
      dst[i] = a[i] / b;
}

inline void maximum_f32_neon_scalar(float *__restrict dst,
                                    const float *__restrict a, const float b,
                                    std::size_t n) {
   for (std::size_t i = 0; i < n; ++i)
      dst[i] = a[i] > b ? a[i] : b;
}

#endif // VEC_FALLBACK_HPP
