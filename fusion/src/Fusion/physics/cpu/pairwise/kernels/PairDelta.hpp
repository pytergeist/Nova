#ifndef FUSION_PHYSICS_CPU_PAIRWISE_KERNELS_PAIR_DELTA_H_
#define FUSION_PHYSICS_CPU_PAIRWISE_KERNELS_PAIR_DELTA_H_

template <class Backend> struct PairDeltaKernel {
   using B = Backend;
   using vec = typename B::vec;

   struct Result {
      vec dx, dy, dz;
   };

   inline Result operator()(vec xi, vec yi, vec zi, vec xj, vec yj,
                            vec zj) const {
      return {B::sub(xi, xj), B::sub(yi, yj), B::sub(zi, zj)};
   }
};

#endif // FUSION_PHYSICS_CPU_PAIRWISE_KERNELS_PAIR_DELTA_H_