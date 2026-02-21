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
void vec3_rowwise_crs(const Vec3Ptrs<T> &pos, const PairCRS &crs, T *out,
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
void block_rowwise_crs(const ParticleT &particles, const PairBlockedCRS &crs,
                       T *out, std::uint64_t E) {
   using B = Neon128<T>;
   using vec = B::vec;

   constexpr std::size_t TILE = ParticleT::tile();
   static_assert(TILE > 0);

   T *out_x = out + 0 * E;
   T *out_y = out + 1 * E;
   T *out_z = out + 2 * E;

   for (std::size_t ib = 0; ib < particles.nBlocks(); ++ib) {
      const std::size_t valid = particles.valid_in_block(ib);
      if (valid == 0)
         continue;

      auto Xi = B::load(particles.x_block_ptr(0, ib));
      auto Yi = B::load(particles.x_block_ptr(1, ib));
      auto Zi = B::load(particles.x_block_ptr(2, ib));

      const std::uint32_t gk = crs.ib_ptr[ib];
      const std::uint32_t gk1 = crs.ib_ptr[ib + 1];

      for (std::uint32_t g = gk; g < gk1; ++g) {
         const std::uint32_t jb = crs.jb_idx[g];
         vec Xj = B::load(particles.x_block_ptr(0, jb));
         vec Yj = B::load(particles.x_block_ptr(1, jb));
         vec Zj = B::load(particles.x_block_ptr(2, jb));

         const std::uint32_t jk = crs.jb_ptr[g];
         const std::uint32_t jk1 = crs.jb_ptr[g + 1];

         for (uint32_t k = jk; k + B::kStep - 1 < jk1; k += B::kStep) {

            vec il = B::load_lane_indices(crs.i_lane.data() + k);
            vec jl = B::load_lane_indices(crs.j_lane.data() + k);

            vec xi = B::gather_lanes(Xi, il);
            vec yi = B::gather_lanes(Yi, il);
            vec zi = B::gather_lanes(Zi, il);

            vec xj = B::gather_lanes(Xj, jl);
            vec yj = B::gather_lanes(Yj, jl);
            vec zj = B::gather_lanes(Zj, jl);

            vec dx = B::sub(xi, xj);
            vec dy = B::sub(yi, yj);
            vec dz = B::sub(zi, zj);
            // Invariant: Here we are returning out ptr stored in group order,
            // not in edge order
            // We can do a linear reorder from BCRS indices to edge indices if
            // needed
            B::store(out_x + k, dx);
            B::store(out_y + k, dy);
            B::store(out_z + k, dz);
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