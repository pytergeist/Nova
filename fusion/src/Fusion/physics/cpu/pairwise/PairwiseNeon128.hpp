#ifndef FUSION_PHYSICS_CPU_PAIRWISE_NEON_128
#define FUSION_PHYSICS_CPU_PAIRWISE_NEON_128

#include "backend/PairwiseLoop.hpp"

#include "Fusion/physics/core/State.hpp"

#include "Fusion/cpu/simd/backend/BackendNeon128.hpp"

namespace pairwise {

template <typename T, class ParticlesT>
inline void sub_blocked_crs(const ParticlesT &particles, const BlockedCRS &crs,
                            T *out, std::uint64_t E) {

   using B = Neon128<T>;
   return pairwise::vec3_block_crs_apply<T, ParticlesT, B>(
       particles, crs, out, E,
       [](B::vec vx, B::vec vy) -> B::vec { return B::sub(vx, vy); });
   //       [](T x, T y) -> T { return x + y; }); // TODO: programme fallbacks
}

} // namespace pairwise

#endif // FUSION_PHYSICS_CPU_PAIRWISE_NEON_128