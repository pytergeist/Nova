#include <arm_neon.h>
#include <iostream>
#include <stdint.h>
#include <type_traits>

namespace simd {

std::pair<bool, bool> is_scalar(size_t na, size_t nb) {
  return {na == 1, nb == 1};
}


template <typename T>
void vec128_addition_neon(float *dst, const float *a, const float *b,
                          std::size_t na, std::size_t nb) {
  bool a_is_scalar = (na == 1), b_is_scalar = (nb == 1);

  const std::size_t N = a_is_scalar ? nb : na;

  float32x4_t a_dup, b_dup;
  if (a_is_scalar)
    a_dup = vdupq_n_f32(a[0]);
  if (b_is_scalar)
    b_dup = vdupq_n_f32(b[0]);

  std::size_t i = 0;
  const std::size_t vec_end = (N / 4) * 4;
  for (int i = 0; i < vec_end; i += 4) {
    float32x4_t va = a_is_scalar ? a_dup : vld1q_f32(a + i);
    float32x4_t vb = b_is_scalar ? b_dup : vld1q_f32(b + i);
    float32x4_t vc = vaddq_f32(va, vb);
    vst1q_f32(dst + i, vc);
  }

  for (; i < N; ++i) {
    float fa = a_is_scalar ? a[0] : a[i];
    float fb = b_is_scalar ? b[0] : b[i];
    dst[i] = fa + fb;
  }
}


} // namespace simd
