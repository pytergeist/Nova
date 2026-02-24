#ifndef FUSION_PHYSICS_CPU_PAIRWISE_NEON_128
#define FUSION_PHYSICS_CPU_PAIRWISE_NEON_128

#include "backend/PairwiseLoop.hpp"

#include "fusion/physics/cpu/pairwise/kernels/PairDelta.hpp"
#include "fusion/physics/cpu/pairwise/kernels/PairR2.hpp"

#include "fusion/physics/cpu/pairwise/store/StoragePolicy.hpp"

#include "Fusion/physics/core/State.hpp"
#include "Fusion/physics/cpu/pairwise/kernels/LJEnergy.hpp"
#include "Fusion/physics/potentials/LJ.hpp"

#include "Fusion/cpu/simd/backend/BackendNeon128.hpp"

#if defined(FUSION_ENABLE_NEON) &&                                             \
    (defined(__ARM_NEON) || defined(__ARM_NEON__))

#include <arm_neon.h>
#include <sleef.h>

namespace pairwise {

template <typename T, class ParticlesT>
inline void pair_r2(const ParticlesT &particles, const PairBlockedCRS &crs,
                    T *out, std::uint64_t E) {
   using B = Neon128<T>;
   PairR2Kernel<B> kernel{};
   StoreScalar<T, B> store{};
   pairwise::block_crs_traverse<T, ParticlesT, B>(particles, crs, out, E,
                                                  kernel, store);
}

template <typename T, class ParticlesT>
inline void pair_delta(const ParticlesT &particles, const PairBlockedCRS &crs,
                       T *out, std::uint64_t E) {
   using B = Neon128<T>;
   T *out_x = out + 0 * E;
   T *out_y = out + 1 * E;
   T *out_z = out + 2 * E;

   PairDeltaKernel<B> kernel{};
   StoreDelta3<T, B> store{out_x, out_y, out_z};
   pairwise::block_crs_traverse<T, ParticlesT, B>(particles, crs, out, E,
                                                  kernel, store);
}

template <typename T, class ParticlesT>
inline void lj_energy(const ParticlesT &particles, const PairBlockedCRS &crs,
                      T *out, std::uint64_t E, const LJParams<T> &params) {
   using B = Neon128<T>;
   using vec = typename B::vec;

   LJEnergyKernel<T, B> kernel;
   kernel.p = params;

   auto store = [&](T *outp, std::uint32_t k, vec v) { B::store(outp + k, v); };

   return pairwise::block_crs_traverse<T, ParticlesT, B>(particles, crs, out, E,
                                                         kernel, store);
}

} // namespace pairwise

#else

// TODO: Need to build fallbacks

#endif

#endif // FUSION_PHYSICS_CPU_PAIRWISE_NEON_128