#ifndef FUSION_PHYSICS_VEC3_GATHER_SUB
#define FUSION_PHYSICS_VEC3_GATHER_SUB

#include "Fusion/common/Hints.hpp"
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

template <typename T, class Tag, class ParticleT>
void block_rowwise_crs(const ParticleT &pos, const CRS &crs, T *out,
                            std::uint64_t E) {
   using B = Neon128<T>;
//   static_assert(simd_traits<Tag, T>::available,
//                 "SIMD kernel requires simd_traits<Tag,T>::available == true");

   constexpr std::size_t TILE = ParticleT::tile();
   static_assert(TILE > 0);

   T *out_x = out + 0 * E;
   T *out_y = out + 1 * E;
   T *out_z = out + 2 * E;

   alignas(16) T xi_lane[TILE];
   alignas(16) T yi_lane[TILE];
   alignas(16) T zi_lane[TILE];

   for (std::size_t b = 0; b < pos.nBlocks(); ++b) {
      const std::size_t valid = pos.valid_in_block(b);
      if (valid == 0)
         continue;

      auto Xi = B::load(pos.x_block_ptr(0, b));
      auto Yi = B::load(pos.x_block_ptr(1, b));
      auto Zi = B::load(pos.x_block_ptr(2, b));

      B::store(xi_lane, Xi);
      B::store(yi_lane, Yi);
      B::store(zi_lane, Zi);

      for (std::size_t lane = 0; lane < valid; ++lane) {
         const std::uint32_t i = static_cast<std::uint32_t>(b * TILE + lane);

         const T xi = xi_lane[lane];
         const T yi = yi_lane[lane];
         const T zi = zi_lane[lane];

         const auto start = crs.row_ptr[i];
         const auto end = crs.row_ptr[i + 1];

         for (auto e = start; e < end; ++e) {
            const auto j = crs.col_idx[e];

            out_x[e] = xi - pos.x_at(0, j);
            out_y[e] = yi - pos.x_at(1, j);
            out_z[e] = zi - pos.x_at(2, j);
         }
      }
   }
}


#else

//template <typename T, class ParticleT>
//void block_rowwise_crs(const ParticleT &pos, const CRS &crs, T *out,
//                       std::uint64_t E) {
//   T *out_x = out + 0 * E;
//   T *out_y = out + 1 * E;
//   T *out_z = out + 2 * E;
//
//   for (std::uint32_t i = 0; i < crs.N; ++i) {
//      const T xi = pos.x_at(0, i);
//      const T yi = pos.x_at(1, i);
//      const T zi = pos.x_at(2, i);
//
//      const auto start = crs.row_ptr[i];
//      const auto end = crs.row_ptr[i + 1];
//
//      for (auto e = start; e < end; ++e) {
//         const auto j = crs.col_idx[e];
//
//         out_x[e] = xi - pos.x_at(0, j);
//         out_y[e] = yi - pos.x_at(1, j);
//         out_z[e] = zi - pos.x_at(2, j);
//      }
//   }
//}

#endif


#endif // FUSION_PHYSICS_VEC3_GATHER_SUB