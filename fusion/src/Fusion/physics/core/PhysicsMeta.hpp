#ifndef FUSION_PHYSICS_META_HPP
#define FUSION_PHYSICS_META_HPP

#include <cstddef>

#include "PhysicsPlan.hpp"

template <typename T> struct PairwiseMeta {
   bool fastpath;
   std::size_t fast_len;
   std::vector<std::size_t> out_shape;
   PairwisePlan<T> plan;
   EdgeList edges; // TODO: make this generic
};

template <typename T>
inline PairwiseMeta<T> make_pairwise_meta(const ParticlesSoA<T> &psoa,
                                          const EdgeList &edges) {
   PairwiseMeta<T> meta;
   meta.fastpath = false;
   meta.fast_len = edges.size();
   meta.plan = make_pairwise_plan(psoa, edges);
   meta.out_shape = std::vector<std::size_t>{
       psoa.x.shape()[0], static_cast<std::size_t>(meta.plan.E)};
   meta.edges = edges;
   return meta;
}

#endif // FUSION_PHYSICS_META_HPP