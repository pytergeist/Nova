#ifndef FUSION_PHYSICS_ITER_HPP
#define FUSION_PHYSICS_ITER_HPP

#include <array>
#include <cstddef>
#include <cstdint>

#include "Fusion/cpu/simd/SimdTraits.hpp"

#include "Fusion/physics/cpu/pairwise/PairwiseTraits.hpp"
#include "Fusion/physics/cpu/pairwise/Vec3GatherSub.hpp"

#include "PhysicsPlanMeta.hpp"

namespace fusion::physics::iter {

template <typename T, typename IterPlan, class FnPairwise, class ParticlesT>
void for_each_edge(const IterPlan &plan, const ParticlesT &pos, T *out,
                   FnPairwise fn) {
   for (std::size_t e = 0; e < plan.E; e++) {
      std::uint32_t i = plan.edges.i[e];
      std::uint32_t j = plan.edges.j[e];
      fn(pos, out, e, i, j);
   }
}

template <typename T, class Tag, class TensorT, class ParticlesT>
void pairwise_tag(const PairwiseMeta<T, ParticlesT> &meta, TensorT &out) {

   if constexpr (true) {
      // Use AoSoA block-rowwise SIMD path over CRS
      pairwise_traits<T, Tag, ParticlesT>::can_execute(
          meta.plan.particles, meta.plan.crs, out.get_ptr(), meta.plan.E);
      return;
   }

   //   for_each_edge<T, PairwisePlan<T, ParticlesT>>(
   //       meta.plan, meta.plan.psoa, out.get_ptr(),
   //       [&](const ParticlesT &psoa, T *out, std::uint32_t e, std::uint32_t
   //       i,
   //           std::uint32_t j) {
   //          bool a_scalar = false;
   //          bool b_scalar = false;
   //          if constexpr (simd_traits<Tag, T>::available) {
   //             block_rowwise_crs(psoa, meta.plan.crs, out, meta.plan.E);
   //             return;
   //          }
   //          if constexpr (simd_traits<Tag, T>::available) {
   //             simd_traits<Tag, T>::execute_contiguous(
   //                 y + i, y + j, o + e + 0 * meta.plan.E,
   //                 static_cast<size_t>(1), a_scalar, b_scalar);
   //
   //             simd_traits<Tag, T>::execute_contiguous(
   //                 z + i, z + j, o + e + 1 * meta.plan.E,
   //                 static_cast<size_t>(1), a_scalar, b_scalar);
   //
   //             simd_traits<Tag, T>::execute_contiguous(
   //                 z + i, z + j, o + e + 2 * meta.plan.E,
   //                 static_cast<size_t>(1), a_scalar, b_scalar);
   //
   //             return;
   //          }

   //          return;
   //       });
}

} // namespace fusion::physics::iter

#endif // FUSION_PHYSICS_ITER_HPP