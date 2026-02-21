#ifndef FUSION_PHYSICS_CPU_PAIRWISE_NEON_128
#define FUSION_PHYSICS_CPU_PAIRWISE_NEON_128

#include "backend/PairwiseLoop.hpp"

#include "Fusion/physics/core/State.hpp"

#include "Fusion/cpu/simd/backend/BackendNeon128.hpp"

#if defined(FUSION_ENABLE_NEON) &&                                             \
    (defined(__ARM_NEON) || defined(__ARM_NEON__))

#include <arm_neon.h>
#include <sleef.h>

namespace pairwise {

template <typename T, class ParticlesT>
inline void sub_blocked_crs(const ParticlesT &particles,
                            const PairBlockedCRS &crs, T *out,
                            std::uint64_t E) {
   using B = Neon128<T>;
   using vec = B::vec;
   T *out_x = out + 0 * E;
   T *out_y = out + 1 * E;
   T *out_z = out + 2 * E;
   return pairwise::block_crs_traverse<T, ParticlesT, B>(
       particles, crs, out, E,
       [](vec vx, vec vy) -> B::vec { return B::sub(vx, vy); },
       [&](std::uint32_t k, vec dx, vec dy, vec dz) {
          B::store(out_x + k, dx);
          B::store(out_y + k, dy);
          B::store(out_z + k, dz);
       });
}

template <typename T, class ParticlesT>
inline void r2_blocked_crs(const ParticlesT &particles,
                           const PairBlockedCRS &crs, T *out, std::uint64_t E) {
   using B = Neon128<T>;
   using vec = B::vec;
   return pairwise::block_crs_traverse<T, ParticlesT, B>(
       particles, crs, out, E,
       [](vec vx, vec vy) -> B::vec { return B::sub(vx, vy); },
       [&](std::uint32_t k, vec dx, vec dy, vec dz) {
          B::store(out + k, B::add(B::add(dx * dx, dy * dy), dy * dz));
       });
}

} // namespace pairwise

#else

// Need to programme fallbacks in here

#endif

#endif // FUSION_PHYSICS_CPU_PAIRWISE_NEON_128