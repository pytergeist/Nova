#ifndef NEW_VEC_128_NEON_HPP
#define NEW_VEC_128_NEON_HPP

#include <cstddef>

#if defined(FUSION_ENABLE_NEON) &&                                             \
    (defined(__ARM_NEON) || defined(__ARM_NEON__))
#include <arm_neon.h>
#endif

template <typename T> struct Neon128;

template <> struct Neon128<float> {

   using U = float;
   using vec = float32x4_t;
   using wide_vec = float32x4x4_t;
   using mask = uint32x4_t;

   static constexpr std::size_t kVectorBytes = 16;
   static constexpr std::size_t kLanes = kVectorBytes / sizeof(U);
   static constexpr std::size_t kUnroll = 4;
   static constexpr std::size_t kBlock = kUnroll * kLanes; // 16

   static constexpr std::size_t kStepVec = kBlock;
   static constexpr std::size_t kStep = kUnroll;

   static wide_vec wide_load(const U *x) { return vld1q_f32_x4(x); }
   static vec load(const U *x) { return vld1q_f32(x); }

   static void wide_store(U *dst, wide_vec x) { vst1q_f32_x4(dst, x); }
   static void store(U *dst, vec x) { vst1q_f32(dst, x); }

   // cgt = compare greater than
   // cge = comapre greater than equal
   static mask cgt(vec x, vec y) { return vcgtq_f32(x, y); }
   static mask cge(vec x, vec y) { return vcgeq_f32(x, y); }
   static vec duplicate(U x) { return vdupq_n_f32(x); }

   // below is a bitwise select, the first param is a mask,
   // the second/third are values to choose based on the mask
   // for operators like >/< of float type these will be x = 1.0f, y = 0.0f
   static vec blend(mask m, vec x, vec y) { return vbslq_f32(m, x, y); }

   static vec add(vec x, vec y) { return vaddq_f32(x, y); }
   static vec sub(vec x, vec y) { return vsubq_f32(x, y); }
   static vec mul(vec x, vec y) { return vmulq_f32(x, y); }
   static vec div(vec x, vec y) { return vdivq_f32(x, y); }

   static vec maximum(vec x, vec y) { return vmaxq_f32(x, y); }
   static vec pow(vec x, vec y) { return Sleef_powf4_u10(x, y); }

   static vec sqrt(vec x) { return vsqrtq_f32(x); }
   static vec log(vec x) { return Sleef_logf4_u10(x); }
   static vec exp(vec x) { return Sleef_expf4_u10(x); }
};

#endif // NEW_VEC_128_NEON_HPP
