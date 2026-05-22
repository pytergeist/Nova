#ifndef FUSION_PHYSICS_POTENTIALS_NONBONDED_H
#define FUSION_PHYSICS_POTENTIALS_NONBONDED_H

#include "Fusion/simulation/core/InteractionPlanMeta.hpp"
#include "Fusion/simulation/core/Neighbours.hpp"
#include "Fusion/simulation/core/TopoIter.hpp"
#include "Fusion/simulation/ops/Utils.hpp"

#include "../../core/tensor/RawTensor.hpp"

template <typename T, class ParticlesT>
inline RawTensor<T> lj_energy(ParticlesT &p, EdgeList &edges,
                              LJParams<T> params) {
   // NB: this operation constructs plan/metas each time it is run
   // and is not intended to be in any hot paths
   const std::size_t channels = 1;
   PairwiseMeta<T, ParticlesT> meta =
       make_pairwise_meta<T, ParticlesT>(p, edges, channels);

   RawTensor<T> out = init_out_from_meta(p.x, meta);
   fusion::physics::iter::pairwise_tag<LJEnergy, T>(meta, out, params);
   return out;
}

template <typename T, class ParticlesT>
inline RawTensor<T> lj_force(ParticlesT &p, EdgeList &edges,
                             LJParams<T> params) {
   // NB: this operation constructs plan/metas each time it is run
   // and is not intended to be in any hot paths
   const std::size_t channels = 3;
   PairwiseMeta<T, ParticlesT> meta =
       make_pairwise_meta<T, ParticlesT>(p, edges, channels);

   RawTensor<T> out = init_out_from_meta(p.x, meta);
   fusion::physics::iter::pairwise_tag<LJForce, T>(meta, out, params);
   return out;
}

template <typename T, class ParticlesT>
inline RawTensor<T> lj_energy_from_meta(const RawTensor<T> &x,
                                        const ParticlesT &p,
                                        const PairwiseMeta<T, ParticlesT> &meta,
                                        LJParams<T> params) {
   RawTensor<T> out = init_out_from_meta(x, meta);
   auto pv = make_view_x(p, x);
   fusion::physics::iter::pairwise_tag<LJEnergy, T>(meta, pv, out, params);
   return out;
}

template <typename T, class ParticlesT>
inline RawTensor<T> lj_force_from_meta(const RawTensor<T> &x,
                                       const ParticlesT &p,
                                       const PairwiseMeta<T, ParticlesT> &meta,
                                       LJParams<T> params) {
   RawTensor<T> out = init_out_from_meta(x, meta);
   auto pv = make_view_x(p, x);
   fusion::physics::iter::pairwise_tag<LJForce, T>(meta, pv, out, params);
   return out;
}

#endif // FUSION_PHYSICS_POTENTIALS_NONBONDED_H
