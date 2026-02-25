#ifndef FUSION_PHYSICS_CPU_PAIRWISE_KERNELS_PAIR_R2_H_
#define FUSION_PHYSICS_CPU_PAIRWISE_KERNELS_PAIR_R2_H_

#include "Fusion/cpu/simd/backend/BackendConcept.hpp"


template <BackendConcept Backend> struct PairR2Kernel {
   using B = Backend;
   using vec = typename B::vec;

   inline vec operator()(vec xi, vec yi, vec zi, vec xj, vec yj, vec zj) const {
      vec dx = B::sub(xi, xj);
      vec dy = B::sub(yi, yj);
      vec dz = B::sub(zi, zj);
      return B::add(B::add(B::mul(dx, dx), B::mul(dy, dy)), B::mul(dz, dz));
   }
};
#endif // FUSION_PHYSICS_CPU_PAIRWISE_KERNELS_PAIR_R2_H__