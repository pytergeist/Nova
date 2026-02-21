// PhysicsPlanMeta.hpp
// Umbrella header for physics dispatch/runtime metadata structs (PairwiseMeta,
// ...)

#ifndef FUSION_PHYSICS_META_HPP
#define FUSION_PHYSICS_META_HPP

#include <cstddef>

#include "PhysicsPlan.h"

template <typename T, class ParticlesT> struct PairwiseMeta {
   bool fastpath;
   std::size_t fast_len;
   std::vector<std::size_t> out_shape;
   PairwisePlan<T, ParticlesT> plan;
};

template <typename T, class ParticlesT>
inline PairwiseMeta<T, ParticlesT> make_pairwise_meta(const ParticlesT &psoa,
                                                      EdgeList &edges,
                                                      std::size_t channels) {
   PairwiseMeta<T, ParticlesT> meta;
   meta.fastpath = false;
   meta.fast_len = edges.E();
   meta.plan = make_pairwise_plan<T, ParticlesT>(psoa, edges);
   meta.out_shape = std::vector<std::size_t>{
       channels, static_cast<std::size_t>(meta.plan.E)};
   return meta;
}

#endif // FUSION_PHYSICS_META_HPP