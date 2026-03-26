#ifndef FUSION_PHYSICS_REGISTRY_PARAM_PLAN_HPP
#define FUSION_PHYSICS_REGISTRY_PARAM_PLAN_HPP

#include <memory>

#include "Fusion/simulation/core/InteractionPlan.h"
#include "Fusion/simulation/core/InteractionPlanMeta.hpp"
#include "Fusion/simulation/cpu/pairwise/PairwiseParams.hpp"

template <typename T, class ParticlesT> struct LJParamPlan {
   PairwiseMeta<T, ParticlesT> meta;
   ParticlesT *particles;
   LJParams<T> params;
};

template <typename T, class ParticlesT> struct GINoParamPlan {
   GatherIndexMeta<T, ParticlesT> meta;
   ParticlesT *particles;
   NoParams params;
};

#endif // FUSION_PHYSICS_REGISTRY_PARAM_PLAN_HPP