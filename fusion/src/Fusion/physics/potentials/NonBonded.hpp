#ifndef FUSION_PHYSICS_POTENTIALS_NONBONDED_H
#define FUSION_PHYSICS_POTENTIALS_NONBONDED_H

#include "Fusion/physics/core/Neighbours.hpp"
#include "Fusion/physics/core/PhysicsIter.hpp"
#include "Fusion/physics/core/PhysicsPlanMeta.hpp"

#include "Fusion/core/RawTensor.hpp"

template <typename T, class ParticlesT>
inline RawTensor<T> init_out_from_meta(const RawTensor<T> &x,
                                       const PairwiseMeta<T, ParticlesT> &m) {
   return RawTensor<T>(m.out_shape, x.dtype(), x.device());
}

template <typename T, class ParticlesT>
inline RawTensor<T> lj_energy(ParticlesT &p, EdgeList &edges,
                              LJParams<T> params) {
   // TODO: Remove BlockedCRS building from meta construction?
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
   // TODO: Remove BlockedCRS building from meta construction?
   const std::size_t channels = 3;
   PairwiseMeta<T, ParticlesT> meta =
       make_pairwise_meta<T, ParticlesT>(p, edges, channels);

   RawTensor<T> out = init_out_from_meta(p.x, meta);
   fusion::physics::iter::pairwise_tag<LJForce, T>(meta, out, params);
   return out;
}

#endif // FUSION_PHYSICS_POTENTIALS_NONBONDED_H