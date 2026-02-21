#ifndef FUSION_PHYSICS_CPU_PAIRWISE_PAIRWISE_TRAITS_H
#define FUSION_PHYSICS_CPU_PAIRWISE_PAIRWISE_TRAITS_H

#include "PairwiseNeon128.hpp"
#include "PairwiseTags.hpp"

template <class Tag, typename T, class ParticlesT> struct pairwise_traits {
   static constexpr bool available = false;
};

template <typename T, class ParticlesT>
struct pairwise_traits<T, Vec3GatherSub, ParticlesT> {
   static constexpr bool available = true;

   static void can_execute(const ParticlesT &particles,
                           const PairBlockedCRS &crs, T *out, std::uint64_t E) {
      pairwise::sub_blocked_crs<T, ParticlesT>(particles, crs, out, E);
   }
};

template <typename T, class ParticlesT>
struct pairwise_traits<T, Vec3GatherR2, ParticlesT> {
   static constexpr bool available = true;

   static void can_execute(const ParticlesT &particles,
                           const PairBlockedCRS &crs, T *out, std::uint64_t E) {
      pairwise::r2_blocked_crs<T, ParticlesT>(particles, crs, out, E);
   }
};

#endif // FUSION_PHYSICS_CPU_PAIRWISE_PAIRWISE_TRAITS_H