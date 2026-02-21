#ifndef FUSION_PHYSICS_PRIMITIVES_LJ
#define FUSION_PHYSICS_PRIMITIVES_LJ

#include <cmath>

#include "Fusion/physics/core/PhysicsPlanMeta.hpp"
#include "Fusion/physics/core/State.hpp"

template <typename T> struct LJParams {
   T epsilon;
   T sigma;
};

template <typename T>
inline T pair_r2(const Vec3Ptrs<T> &P, uint32_t i, uint32_t j) {
   const T dx = P.x[i] - P.x[j];
   const T dy = P.y[i] - P.y[j];
   const T dz = P.z[i] - P.z[j];
   return dx * dx + dy * dy + dz * dz;
}

template <typename T> T pairwise_dist(T *xi, T *xj) {
   T dis = displacement(xi, xj);
   T sqdist = std::pow(dis, 2);
   T sqrtdist = std::sqrt(sqdist);
   std::cout << "pariwise dist = " << sqrtdist << std::endl;
   return sqrtdist;
}

template <typename T>
T lennard_jones(const ParticlesSoA<T> &psoa, const EdgeList &pairs,
                const LJParams<T> &params) {

   PairwiseMeta meta = make_pairwise_meta(psoa, pairs);

   const auto P = psoa.vec3();

   const T eps4 = T(4) * params.epsilon;
   const T sig2 = params.sigma * params.sigma;

   T energy = T(0);

   const std::size_t K = pairs.i.size();
   for (std::size_t k = 0; k < K; ++k) {
      const uint32_t i = pairs.i[k];
      const uint32_t j = pairs.j[k];

      const T r2 = pair_r2(P, i, j);

      const T inv_r2 = sig2 / r2;
      const T inv_r6 = inv_r2 * inv_r2 * inv_r2;
      const T inv_r12 = inv_r6 * inv_r6;

      energy += eps4 * (inv_r12 - inv_r6);
   }

   return energy;
}

#endif // FUSION_PHYSICS_PRIMITIVES_LJ
