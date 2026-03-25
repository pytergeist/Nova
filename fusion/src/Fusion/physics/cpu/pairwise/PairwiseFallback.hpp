#ifndef FUSION_PHYSICS_CPU_PAIRWISE_PAIRWISE_FALLBACK_HPP
#define FUSION_PHYSICS_CPU_PAIRWISE_PAIRWISE_FALLBACK_HPP

#include <cstddef>
#include <cstdint>
#include <utility>

#include "Fusion/physics/core/Neighbours.hpp"

template <typename T, class ParticleT, class Kernel,class Store>
void blocked_crs_walk(const ParticleT &particles, const PairBlockedCRS &crs,
                        T *out, std::uint64_t E, Kernel &&kernel,
                        Store &&store) {


   constexpr std::size_t TILE = ParticleT::tile();
   static_assert(TILE > 0);

   for (std::size_t ib = 0; ib < particles.nBlocks(); ++ib) {
      const std::size_t valid = particles.valid_in_block(ib);
      if (valid == 0)
         continue;

      auto Xi = particles.x_block_ptr(0, ib);
      auto Yi = particles.x_block_ptr(1, ib);
      auto Zi = particles.x_block_ptr(2, ib);

      const std::uint32_t gk = crs.ib_ptr[ib];
      const std::uint32_t gk1 = crs.ib_ptr[ib + 1];

      for (std::uint32_t g = gk; g < gk1; ++g) {
         const std::uint32_t jb = crs.jb_idx[g];
         T* Xj = particles.x_block_ptr(0, jb);
         T* Yj = particles.x_block_ptr(1, jb);
         T* Zj = particles.x_block_ptr(2, jb);

         const std::uint32_t jk = crs.jb_ptr[g];
         const std::uint32_t jk1 = crs.jb_ptr[g + 1];

         for (uint32_t k = jk; k < jk1; k++) {

            T* il = crs.i_lane.data() + k;
            T* jl = crs.j_lane.data() + k;

            T* xi = Xi[il];
            T* yi = Yi[il];
            T* zi = Zi[il];

            T* xj = Xj[jl];
            T* yj = Yj[jl];
            T* zj = Zj[jl];

            auto result = std::forward<Kernel>(kernel)(xi, yi, zi, xj, yj, zj);
            // Invariant: Here we are returning out ptr stored in group order,
            // not in edge order
            // We can do a linear reorder from BCRS indices to edge indices if
            // needed
            store(out, k, result);
         }
      }
   }
}

#endif // FUSION_PHYSICS_CPU_PAIRWISE_PAIRWISE_FALLBACK_HPP