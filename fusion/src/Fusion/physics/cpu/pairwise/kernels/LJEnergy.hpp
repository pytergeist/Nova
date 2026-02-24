#ifndef FUSION_PHYSICS_CPU_PAIRWISE_FUSED_LJ
#define FUSION_PHYSICS_CPU_PAIRWISE_FUSED_LJ

#include "Fusion/cpu/simd/backend/BackendNeon128.hpp"
#include "Fusion/physics/potentials/LJ.hpp"

template <typename T, BackendConcept Backend> struct LJEnergyKernel {

   using B = Backend;
   using vec = typename B::vec;

   LJParams<T> p;

   inline vec operator()(vec xi, vec yi, vec zi, vec xj, vec yj, vec zj) {
      vec dx = B::sub(xi, xj);
      vec dy = B::sub(yi, yj);
      vec dz = B::sub(zi, zj);

      vec r2 = B::add(B::add(B::mul(dx, dx), B::mul(dy, dy)), B::mul(dz, dz));
      vec inv_r2 = B::reciprocal(r2);

      vec epsilon = B::duplicate(p.epsilon);
      vec sigma = B::duplicate(p.sigma);

      vec sigma2 = B::mul(sigma, sigma);
      vec sr2 = B::mul(inv_r2, sigma2);
      vec sr6 = B::pow(sr2, B::duplicate(3.0F));
      vec sr12 = B::mul(sr6, sr6);

      vec ep4 = B::mul(B::duplicate(4.0F), epsilon);
      return B::mul(ep4, B::sub(sr12, sr6));
   }
};

#endif // FUSION_PHYSICS_CPU_PAIRWISE_FUSED_LJ