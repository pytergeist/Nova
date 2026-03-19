#ifndef FUSION_PHYSICS_OPS_UTILS_HPP
#define FUSION_PHYSICS_OPS_UTILS_HPP

#include "Fusion/core/RawTensor.hpp"
#include "Fusion/physics/core/PhysicsPlanMeta.hpp"

template <typename T, class ParticlesT>
RawTensor<T> init_out_from_meta(const RawTensor<T> &x,
                                const PairwiseMeta<T, ParticlesT> &m) {
   return RawTensor<T>(m.plan.out_shape, x.dtype(), x.device());
}

template <typename T, class ParticlesT>
RawTensor<T> init_out_from_meta(const RawTensor<T> &x,
                                const GatherIndexMeta<T, ParticlesT> &m) {
   return RawTensor<T>(m.plan.out_shape, x.dtype(), x.device());
}

#endif // FUSION_PHYSICS_OPS_UTILS_HPP
