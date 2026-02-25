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
      vec sr6 =
          B::pow(sr2, B::duplicate(3.0F)); // TODO: remove pow op from here
      vec sr12 = B::mul(sr6, sr6);

      vec ep4 = B::mul(B::duplicate(4.0F), epsilon);
      return B::mul(ep4, B::sub(sr12, sr6));
   }
};

template <typename T, BackendConcept Backend> struct LJForceKernel {
   using B = Backend;
   using vec = typename B::vec;

   LJParams<T> p;

   struct Force3 {
      vec fx, fy, fz;
   };

   inline Force3 operator()(vec xi, vec yi, vec zi, vec xj, vec yj, vec zj) {
      vec dx = B::sub(xi, xj);
      vec dy = B::sub(yi, yj);
      vec dz = B::sub(zi, zj);

      vec r2 = B::add(B::add(B::mul(dx, dx), B::mul(dy, dy)), B::mul(dz, dz));
      vec inv_r2 = B::reciprocal(r2);

      vec eps = B::duplicate(p.epsilon);
      vec sig = B::duplicate(p.sigma);

      vec sig2 = B::mul(sig, sig);
      vec sr2 = B::mul(sig2, inv_r2);
      vec sr4 = B::mul(sr2, sr2);
      vec sr6 = B::mul(sr4, sr2);
      vec sr12 = B::mul(sr6, sr6);

      vec two_sr12_minus_sr6 = B::sub(B::mul(B::duplicate((T)2), sr12), sr6);
      vec f_over_r = B::mul(B::mul(B::duplicate((T)24), eps),
                            B::mul(inv_r2, two_sr12_minus_sr6));

      return {B::mul(f_over_r, dx), B::mul(f_over_r, dy), B::mul(f_over_r, dz)};
   }
};

#endif // FUSION_PHYSICS_CPU_PAIRWISE_FUSED_LJ