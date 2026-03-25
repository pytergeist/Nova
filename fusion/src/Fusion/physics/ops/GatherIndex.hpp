#ifndef FUSION_PHYSICS_OPS_GATHER_INDEX_HPP
#define FUSION_PHYSICS_OPS_GATHER_INDEX_HPP

#include "Fusion/core/RawTensor.hpp"
#include "Fusion/physics/core/PhysicsIter.hpp"
#include "Fusion/physics/core/PhysicsPlanMeta.hpp"

template <typename T, class ParticlesT>
RawTensor<T>
pair_delta_from_meta(const RawTensor<T> &x, const ParticlesT &p,
                     const GatherIndexMeta<T, ParticlesT> &meta,
                     NoParams params) {
   RawTensor<T> out = init_out_from_meta(x, meta);
   auto pv = make_view_x(p, x);
   fusion::physics::iter::gather_index_tag<PairDelta3, T>(meta, pv, out,
                                                          params);
   return out;
}

#endif // FUSION_PHYSICS_OPS_GATHER_INDEX_HPP
