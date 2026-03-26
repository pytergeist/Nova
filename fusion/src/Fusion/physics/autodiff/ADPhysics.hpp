#ifndef FUSION_PHYSICS_AUTODIFF_AD_PHYSICS_HPP
#define FUSION_PHYSICS_AUTODIFF_AD_PHYSICS_HPP

#include "Fusion/autodiff/ADTensor.hpp"

#include "Fusion/core/RawTensor.hpp"

#include "Fusion/physics/core/PhysicsPlan.h"
#include "Fusion/physics/core/PhysicsPlanMeta.hpp"
#include "Fusion/physics/cpu/pairwise/PairwiseParams.hpp"
#include "ParamPlan.hpp"
#include "registry/pairwise/LJ.hpp"
#include "registry/pairwise/PairDelta3.hpp"

template <typename T, class ParticlesT>
ADTensor<T> lj_energy(ADTensor<T> &x, ParticlesT &particles,
                      PairwiseMeta<T, ParticlesT> &meta, LJParams<T> params) {
   using Meta = PairwiseMeta<T, ParticlesT>;
   using Param = LJParamPlan<T, ParticlesT>;
   Param pplan{meta, &particles, params}; // TODO: evaluate this
   return x.template unary_meta_hook<LennardJones<T, ParticlesT>, Meta, Param>(
       meta, pplan,
       [&](const RawTensor<T> &xb, Meta &meta_, const Param &plan_) {
          return lj_energy_from_meta<T, ParticlesT>(xb, particles, meta_,
                                                    plan_.params);
       });
}

template <typename T, class ParticlesT>
ADTensor<T> pair_delta3(ADTensor<T> &x, ParticlesT &particles,
                        GatherIndexMeta<T, ParticlesT> &meta, NoParams params) {
   using Meta = GatherIndexMeta<T, ParticlesT>;
   using Param = GINoParamPlan<T, ParticlesT>;
   Param pplan{meta, &particles, params};
   return x.template unary_meta_hook<PairDelta3<T, ParticlesT>, Meta, Param>(
       meta, pplan,
       [&](const RawTensor<T> &xb, Meta &meta_, const Param &plan_) {
          return pair_delta3_from_meta<T, ParticlesT>(xb, particles, meta_,
                                                      plan_.params);
       });
}
#endif // FUSION_PHYSICS_AUTODIFF_AD_PHYSICS_HPP
