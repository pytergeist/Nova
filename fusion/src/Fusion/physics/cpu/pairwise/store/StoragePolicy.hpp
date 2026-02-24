#ifndef FUSION_PHYSICS_CPU_PAIRWISE_STORE_STORAGE_POLICY
#define FUSION_PHYSICS_CPU_PAIRWISE_STORE_STORAGE_POLICY

#include <cstdint>

#include "Fusion/physics/cpu/pairwise/kernels/PairDelta.hpp"
#include "Fusion/physics/cpu/pairwise/kernels/PairR2.hpp"

#include "Fusion/cpu/simd/backend/BackendConcept.hpp"

template <class T, BackendConcept Backend> struct StoreScalar {
   using B = Backend;
   using vec = typename B::vec;

   inline void operator()(T *out, std::uint32_t k, vec v) const {
      B::store(out + k, v);
   }
};

template <class T, BackendConcept Backend> struct StoreDelta3 {
   using B = Backend;
   using vec = typename B::vec;
   using Result = typename PairDeltaKernel<Backend>::Result;

   T *out_x;
   T *out_y;
   T *out_z;

   inline void operator()(T * /*out*/, std::uint32_t k, const Result &r) const {
      B::store(out_x + k, r.dx);
      B::store(out_y + k, r.dy);
      B::store(out_z + k, r.dz);
   }
};

#endif // FUSION_PHYSICS_CPU_PAIRWISE_STORE_STORAGE_POLICY