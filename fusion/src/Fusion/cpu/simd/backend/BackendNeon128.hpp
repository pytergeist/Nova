#ifndef FUSION_CPU_NEON128_BACKEND_HPP
#define FUSION_CPU_NEON128_BACKEND_HPP

#include <array>
#include <cstddef>
#include <cstdint>

#if defined(FUSION_ENABLE_NEON) &&                                             \
    (defined(__ARM_NEON) || defined(__ARM_NEON__))
#include <arm_neon.h>
#endif

template <typename T> struct Neon128;

template <> struct Neon128<float> {

   using U = float;
   using vec = float32x4_t;
   using wide_vec = float32x4x4_t;
   using lane_index = uint8x16_t;
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

   static float horizontal_add(vec x) {
#if defined(__aarch64__)
      return vaddvq_f32(x);
#else
      float32x2_t s2 = vadd_f32(vget_low_f32(x), vget_high_f32(x));
      s2 = vpadd_f32(s2, s2);
      return vget_lane_f32(s2, 0);
#endif
   }

   static uint8_t byte_index(uint8_t lane, uint8_t byte) {
      return static_cast<uint8_t>((4U * lane) + byte);
   }

   static float32x4_t load_lane_indices(const std::uint16_t *lanes) {

      const uint8_t l0 = static_cast<uint8_t>(lanes[0]);
      const uint8_t l1 = static_cast<uint8_t>(lanes[1]);
      const uint8_t l2 = static_cast<uint8_t>(lanes[2]);
      const uint8_t l3 = static_cast<uint8_t>(lanes[3]);

      const std::array<uint8_t, 16> idx_arr = {
          byte_index(l0, 0), byte_index(l0, 1), byte_index(l0, 2),
          byte_index(l0, 3), byte_index(l1, 0), byte_index(l1, 1),
          byte_index(l1, 2), byte_index(l1, 3), byte_index(l2, 0),
          byte_index(l2, 1), byte_index(l2, 2), byte_index(l2, 3),
          byte_index(l3, 0), byte_index(l3, 1), byte_index(l3, 2),
          byte_index(l3, 3),
      };

      return vld1q_u8(idx_arr.data());
   }

   static vec gather_lanes(vec v, lane_index idx) {
      const uint8x16_t table = vreinterpretq_u8_f32(v);
      return vreinterpretq_f32_u8(vqtbl1q_u8(table, idx));
   }
};

#endif // FUSION_CPU_NEON128_BACKEND_HPP
