#ifndef FUSION_PHYSICS_ITER_HPP
#define FUSION_PHYSICS_ITER_HPP

#include <array>
#include <cstddef>
#include <cstdint>

#include "Fusion/cpu/simd/SimdTraits.hpp"

#include "Fusion/physics/cpu/kernels/Vec3GatherSub.hpp"

#include "PhysicsMeta.hpp"

namespace fusion::physics::iter {

template <typename T, typename IterPlan, class FnPairwise>
void for_each_edge(const IterPlan &plan, const Vec3Ptrs<T> &pos, T *out,
                   FnPairwise fn) {
   for (std::size_t e = 0; e < plan.E; e++) {
      std::uint32_t i = plan.edges.i[e];
      std::uint32_t j = plan.edges.j[e];
      fn(pos, out, e, i, j);
   }
}

template <typename T, class Tag, class TensorT>
void pairwise_tag(const PairwiseMeta<T> &meta, TensorT &out) {

   const Vec3Ptrs vec3 = meta.plan.psoa.vec3();
   if (meta.fastpath) {
      // TODO: wtf would fast path need here?
   }

   for_each_edge<T, PairwisePlan<T>>(
       meta.plan, vec3, out.get_ptr(),
       [&](const Vec3Ptrs<T> &vec3, T *out, std::uint32_t e, std::uint32_t i,
           std::uint32_t j) {
          bool a_scalar = false;
          bool b_scalar = false;
          if constexpr (simd_traits<Tag, T>::available) {
             vec3_rowwise_crs(vec3, meta.plan.crs, out, meta.plan.E);
             return;
          }
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

          return;
       });
}

} // namespace fusion::physics::iter

#endif // FUSION_PHYSICS_ITER_HPP