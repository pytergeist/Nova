#ifndef FUSION_PHYSICS_ITER_HPP
#define FUSION_PHYSICS_ITER_HPP

#include <array>
#include <cstddef>
#include <cstdint>

#include "Fusion/cpu/simd/SimdTraits.hpp"

#include "PhysicsMeta.hpp"

namespace fusion::physics::iter {

template <typename T, typename IterPlan, std::size_t N, class FnPairwise>
void for_each_edge(const IterPlan &plan, std::array<uint8_t *, N> &base,
                   FnPairwise fn) {
   for (std::size_t e = 0; e < plan.E; e++) {
      std::uint32_t i = plan.edges.i[e];
      std::uint32_t j = plan.edges.j[e];
      fn(base, e, i, j);
   }
}

template <typename T, class Tag, class TensorT>
void pairwise_tag(const PairwiseMeta<T> &meta, TensorT &out) {

   const Vec3Ptrs vec3 = meta.plan.psoa.vec3();
   std::array<uint8_t *, 4> base = {
       reinterpret_cast<uint8_t *>(const_cast<T *>(out.get_ptr())),
       reinterpret_cast<uint8_t *>(const_cast<T *>(vec3.x)),
       reinterpret_cast<uint8_t *>(const_cast<T *>(vec3.y)),
       reinterpret_cast<uint8_t *>(const_cast<T *>(vec3.z))};

   if (meta.fastpath) {
      // TODO: wtf would fast path need here?
   }

   for_each_edge<T, PairwisePlan<T>, 4>(
       meta.plan, base,
       [&](std::array<uint8_t *, 4> &base, std::uint32_t e, std::uint32_t i,
           std::uint32_t j) {
          auto *o = reinterpret_cast<T *>(base[0]);
          auto *x = reinterpret_cast<T *>(base[1]);
          auto *y = reinterpret_cast<T *>(base[2]);
          auto *z = reinterpret_cast<T *>(base[3]);
          bool a_scalar = false;
          bool b_scalar = false;
          if constexpr (simd_traits<Tag, T>::available) {
             simd_traits<Tag, T>::execute_contiguous(
                 y + i, y + j, o + e + 0 * meta.plan.E, static_cast<size_t>(1),
                 a_scalar, b_scalar);

             simd_traits<Tag, T>::execute_contiguous(
                 z + i, z + j, o + e + 1 * meta.plan.E, static_cast<size_t>(1),
                 a_scalar, b_scalar);

             simd_traits<Tag, T>::execute_contiguous(
                 z + i, z + j, o + e + 2 * meta.plan.E, static_cast<size_t>(1),
                 a_scalar, b_scalar);

             return;
          }

          return;
       });
}

} // namespace fusion::physics::iter

#endif // FUSION_PHYSICS_ITER_HPP