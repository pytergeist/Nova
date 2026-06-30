#ifndef FUSION_OPS_INDEXED_PAIR_VECTOR_HPP
#define FUSION_OPS_INDEXED_PAIR_VECTOR_HPP

#include "Fusion/core/tensor/AoSoATensor.hpp"

template <typename T, class ParticlesT>
AoSoATensor<T> pair_delta3_from_meta(const AoSoATensor<T> &x, ParticlesT &p,
                                     const GatherIndexMeta<T, ParticlesT> &meta,
                                     NoParams params) {
   AoSoATensor<T> out = init_out_from_meta(x, meta);
   auto pv = make_view_x(p, x);
   fusion::physics::iter::gather_index_tag<PairDelta3SIMD, T>(meta, pv, out,
                                                              params);
   return out;
}

#endif // FUSION_OPS_INDEXED_PAIR_VECTOR_HPP