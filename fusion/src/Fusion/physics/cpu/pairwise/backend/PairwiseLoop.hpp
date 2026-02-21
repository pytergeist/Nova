#ifndef FUSION_PHYSICS_CPU_PAIRWISE_LOOP_HPP
#define FUSION_PHYSICS_CPU_PAIRWISE_LOOP_HPP

#include <cstddef>
#include <cstdint>

#include "Fusion/physics/core/Neighbours.hpp"
#include "Fusion/physics/core/State.hpp"

#include "Fusion/cpu/simd/backend/BackendConcept.hpp"
#include "Fusion/cpu/simd/backend/BackendNeon128.hpp"

namespace pairwise {

template <typename T, class ParticleT, BackendConcept Backend,
          class BinaryVecOp, class Consumer>
void block_crs_traverse(const ParticleT &particles, const PairBlockedCRS &crs,
                        T *out, std::uint64_t E, BinaryVecOp &&op,
                        Consumer &&consumer) {
   using B = Neon128<T>;
   using vec = B::vec;

   constexpr std::size_t TILE = ParticleT::tile();
   static_assert(TILE > 0);

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

            vec dx = op(xi, xj);
            vec dy = op(yi, yj);
            vec dz = op(zi, zj);
            // Invariant: Here we are returning out ptr stored in group order,
            // not in edge order
            // We can do a linear reorder from BCRS indices to edge indices if
            // needed
            std::forward<Consumer>(consumer)(k, dx, dy, dz);
         }
      }
   }
}

} // namespace pairwise

#endif // FUSION_PHYSICS_CPU_PAIRWISE_LOOP_HPP