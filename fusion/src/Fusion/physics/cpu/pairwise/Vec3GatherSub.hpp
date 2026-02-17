#ifndef FUSION_PHYSICS_VEC3_GATHER_SUB
#define FUSION_PHYSICS_VEC3_GATHER_SUB

#include "Fusion/common/Hints.hpp"
#include "Fusion/physics/core/Neighbours.hpp"
#include "Fusion/physics/core/State.hpp"

#if defined(FUSION_ENABLE_NEON) &&                                             \
    (defined(__ARM_NEON) || defined(__ARM_NEON__))

#include <arm_neon.h>
#include <sleef.h>

#include "Fusion/cpu/simd/backend/BackendNeon128.hpp"

template <typename T>
void vec3_rowwise_crs(const Vec3Ptrs<T> &pos, const CRS &crs, T *out,
                      std::uint64_t E) {
   T *out_x = out + 0 * E;
   T *out_y = out + 1 * E;
   T *out_z = out + 2 * E;

   std::uint64_t e_global = 0;

   for (std::uint32_t i = 0; i < crs.N; ++i) {
      const T xi = pos.x[i];
      const T yi = pos.y[i];
      const T zi = pos.z[i];

      const auto start = crs.row_ptr[i];
      const auto end = crs.row_ptr[i + 1];

      for (auto p = start; p < end; ++p, ++e_global) {
         const auto j = crs.col_idx[p];

         out_x[e_global] = xi - pos.x[j];
         out_y[e_global] = yi - pos.y[j];
         out_z[e_global] = zi - pos.z[j];
      }
   }
}

template <class Backend>
static Backend::vec gather_lanes(typename Backend::vec v, uint8_t l0,
                                 uint8_t l1, uint8_t l2, uint8_t l3) {
   const uint8x16_t table = vreinterpretq_u8_f32(v);
   const uint8x16_t idx = (uint8x16_t){
       uint8_t(4 * l0 + 0), uint8_t(4 * l0 + 1), uint8_t(4 * l0 + 2),
       uint8_t(4 * l0 + 3), uint8_t(4 * l1 + 0), uint8_t(4 * l1 + 1),
       uint8_t(4 * l1 + 2), uint8_t(4 * l1 + 3), uint8_t(4 * l2 + 0),
       uint8_t(4 * l2 + 1), uint8_t(4 * l2 + 2), uint8_t(4 * l2 + 3),
       uint8_t(4 * l3 + 0), uint8_t(4 * l3 + 1), uint8_t(4 * l3 + 2),
       uint8_t(4 * l3 + 3)};
   return vreinterpretq_f32_u8(vqtbl1q_u8(table, idx));
}

template <typename T, class Tag, class ParticleT>
void block_rowwise_crs(const ParticleT &particles, const BlockedCRS &crs,
                       T *out, std::uint64_t E) {
   using B = Neon128<T>;
   //   static_assert(simd_traits<Tag, T>::available,
   //                 "SIMD kernel requires simd_traits<Tag,T>::available ==
   //                 true");

   constexpr std::size_t TILE = ParticleT::tile();
   static_assert(TILE > 0);

   T *out_x = out + 0 * E;
   T *out_y = out + 1 * E;
   T *out_z = out + 2 * E;

   alignas(16) T xi[TILE], yi[TILE], zi[TILE];
   alignas(16) T xj[TILE], yj[TILE], zj[TILE];

   for (std::size_t ib = 0; ib < particles.nBlocks(); ++ib) {
      const std::size_t valid = particles.valid_in_block(ib);
      if (valid == 0)
         continue;

      auto Xi = B::load(particles.x_block_ptr(0, ib));
      auto Yi = B::load(particles.x_block_ptr(1, ib));
      auto Zi = B::load(particles.x_block_ptr(2, ib));

      B::store(xi, Xi);
      B::store(yi, Yi);
      B::store(zi, Zi);

      const std::uint32_t gk = crs.ib_ptr[ib];
      const std::uint32_t gk1 = crs.ib_ptr[ib + 1];

      for (std::uint32_t g = gk; g < gk1; ++g) {
         const std::uint32_t jb = crs.jb_idx[g];
         auto Xj = B::load(particles.x_block_ptr(0, jb));
         auto Yj = B::load(particles.x_block_ptr(1, jb));
         auto Zj = B::load(particles.x_block_ptr(2, jb));

         B::store(xj, Xj);
         B::store(yj, Yj);
         B::store(zj, Zj);

         const std::uint32_t jk = crs.jb_ptr[g];
         const std::uint32_t jk1 = crs.jb_ptr[g + 1];

         for (uint32_t k = jk; k + 3 < jk1; k += 4) {
            uint8_t il0 = (uint8_t)crs.i_lane[k + 0];
            uint8_t il1 = (uint8_t)crs.i_lane[k + 1];
            uint8_t il2 = (uint8_t)crs.i_lane[k + 2];
            uint8_t il3 = (uint8_t)crs.i_lane[k + 3];

            uint8_t jl0 = (uint8_t)crs.j_lane[k + 0];
            uint8_t jl1 = (uint8_t)crs.j_lane[k + 1];
            uint8_t jl2 = (uint8_t)crs.j_lane[k + 2];
            uint8_t jl3 = (uint8_t)crs.j_lane[k + 3];

            auto xi4 = gather_lanes<B>(Xi, il0, il1, il2, il3);
            auto yi4 = gather_lanes<B>(Yi, il0, il1, il2, il3);
            auto zi4 = gather_lanes<B>(Zi, il0, il1, il2, il3);

            auto xj4 = gather_lanes<B>(Xj, jl0, jl1, jl2, jl3);
            auto yj4 = gather_lanes<B>(Yj, jl0, jl1, jl2, jl3);
            auto zj4 = gather_lanes<B>(Zj, jl0, jl1, jl2, jl3);

            auto dx4 = B::sub(xi4, xj4);
            auto dy4 = B::sub(yi4, yj4);
            auto dz4 = B::sub(zi4, zj4);

            alignas(16) float dx[4], dy[4], dz[4];
            B::store(dx, dx4);
            B::store(dy, dy4);
            B::store(dz, dz4);

            uint32_t e0 = crs.e_idx[k + 0];
            uint32_t e1 = crs.e_idx[k + 1];
            uint32_t e2 = crs.e_idx[k + 2];
            uint32_t e3 = crs.e_idx[k + 3];

            out_x[e0] = dx[0];
            out_y[e0] = dy[0];
            out_z[e0] = dz[0];
            out_x[e1] = dx[1];
            out_y[e1] = dy[1];
            out_z[e1] = dz[1];
            out_x[e2] = dx[2];
            out_y[e2] = dy[2];
            out_z[e2] = dz[2];
            out_x[e3] = dx[3];
            out_y[e3] = dy[3];
            out_z[e3] = dz[3];
         }
      }
   }
}

#else

// template <typename T, class ParticleT>
// void block_rowwise_crs(const ParticleT &pos, const CRS &crs, T *out,
//                        std::uint64_t E) {
//    T *out_x = out + 0 * E;
//    T *out_y = out + 1 * E;
//    T *out_z = out + 2 * E;
//
//    for (std::uint32_t i = 0; i < crs.N; ++i) {
//       const T xi = pos.x_at(0, i);
//       const T yi = pos.x_at(1, i);
//       const T zi = pos.x_at(2, i);
//
//       const auto start = crs.row_ptr[i];
//       const auto end = crs.row_ptr[i + 1];
//
//       for (auto e = start; e < end; ++e) {
//          const auto j = crs.col_idx[e];
//
//          out_x[e] = xi - pos.x_at(0, j);
//          out_y[e] = yi - pos.x_at(1, j);
//          out_z[e] = zi - pos.x_at(2, j);
//       }
//    }
// }

#endif

#endif // FUSION_PHYSICS_VEC3_GATHER_SUB