#ifndef FUSION_PHYSICS_CPU_PAIRWISE_PAIRWISE_TRAITS_H
#define FUSION_PHYSICS_CPU_PAIRWISE_PAIRWISE_TRAITS_H

#include "PairwiseNeon128.hpp"
#include "PairwiseParams.hpp"
#include "PairwiseTags.hpp"

template <class Tag, typename T, class ParticlesT> struct pairwise_traits {
   static constexpr bool available = false;
};

template <typename T, class ParticlesT>
struct pairwise_traits<PairDelta3SIMD, T, ParticlesT> {
   static constexpr bool available = true;

   static void can_execute(const ParticlesT &particles,
                           const PairBlockedCRS &crs, T *out, std::uint64_t E,
                           params_type_t<PairDelta3SIMD, T> = {}) {
      pairwise::pair_delta<T, ParticlesT>(particles, crs, out, E);
   }
};

template <typename T, class ParticlesT>
struct pairwise_traits<PairR2SIMD, T, ParticlesT> {
   static constexpr bool available = true;

   static void can_execute(const ParticlesT &particles,
                           const PairBlockedCRS &crs, T *out, std::uint64_t E,
                           params_type_t<PairDelta3SIMD, T> = {}) {
      pairwise::pair_r2<T, ParticlesT>(particles, crs, out, E);
   }
};

template <typename T, class ParticlesT>
struct pairwise_traits<LJEnergy, T, ParticlesT> {
   static constexpr bool available = true;

   static void can_execute(const ParticlesT &particles,
                           const PairBlockedCRS &crs, T *out, std::uint64_t E,
                           const LJParams<T> &params) {
      pairwise::lj_energy<T, ParticlesT>(particles, crs, out, E, params);
   }
};

template <typename T, class ParticlesT>
struct pairwise_traits<LJForce, T, ParticlesT> {
   static constexpr bool available = true;

   static void can_execute(const ParticlesT &particles,
                           const PairBlockedCRS &crs, T *out, std::uint64_t E,
                           const LJParams<T> &params) {
      pairwise::lj_force<T, ParticlesT>(particles, crs, out, E, params);
   }
};

#endif // FUSION_PHYSICS_CPU_PAIRWISE_PAIRWISE_TRAITS_H