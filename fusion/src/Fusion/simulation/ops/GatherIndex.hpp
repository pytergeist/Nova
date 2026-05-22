#ifndef FUSION_PHYSICS_OPS_GATHER_INDEX_HPP
#define FUSION_PHYSICS_OPS_GATHER_INDEX_HPP

#include "../../core/tensor/RawTensor.hpp"
#include "Fusion/simulation/core/InteractionPlanMeta.hpp"
#include "Fusion/simulation/core/TopoIter.hpp"

template <typename T, class ParticlesT>
RawTensor<T> pair_delta3_from_meta(const RawTensor<T> &x, ParticlesT &p,
                                   const GatherIndexMeta<T, ParticlesT> &meta,
                                   NoParams params) {
   RawTensor<T> out = init_out_from_meta(x, meta);
   auto pv = make_view_x(p, x);
   fusion::physics::iter::gather_index_tag<PairDelta3SIMD, T>(meta, pv, out,
                                                              params);
   return out;
}

#endif // FUSION_PHYSICS_OPS_GATHER_INDEX_HPP
