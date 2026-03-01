#ifndef FUSION_PHYSICS_AUTODIFF_AD_PHYSICS_HPP
#define FUSION_PHYSICS_AUTODIFF_AD_PHYSICS_HPP

#include "Fusion/autodiff/ADTensor.hpp"

#include "Fusion/core/RawTensor.hpp"

#include "Fusion/physics/core/PhysicsPlanMeta.hpp"
#include "Fusion/physics/cpu/pairwise/PairwiseParams.hpp"
#include "ParamPlan.hpp"
#include "registry/pairwise/LJ.hpp"

template <typename T, class ParticlesT>
ADTensor<T> lj_energy(ADTensor<T> &x, ParticlesT &particles,
                      PairwiseMeta<T, ParticlesT> &meta, LJParams<T> params) {
   using Meta = PairwiseMeta<T, ParticlesT>;
   using Param = LJParamPlan<T, ParticlesT>;
   Param pplan{meta, &particles, params};
   return x.template unary_meta_hook<LennardJones<T, ParticlesT>, Meta, Param>(
       meta, pplan,
       [&](const RawTensor<T> &xb, Meta &meta_, const Param &plan_) {
          return lj_energy_from_meta<T, ParticlesT>(xb, particles, meta_,
                                                    plan_.params);
       });
}

#endif // FUSION_PHYSICS_AUTODIFF_AD_PHYSICS_HPP