#ifndef FUSION_CORE_ITER_TOPO_ITER_HPP
#define FUSION_CORE_ITER_TOPO_ITER_HPP

#include <cstddef>

#include "Fusion/simulation/cpu/pairwise/PairwiseParams.hpp"
#include "Fusion/simulation/cpu/pairwise/PairwiseTraits.hpp"

#include "InteractionPlanMeta.hpp"

namespace fusion::topo::iter {

template <typename T, typename IterPlan, class FnPairwise, class ParticlesT>
void for_each_edge(const IterPlan &plan, const ParticlesT &pos, T *out,
                   FnPairwise fn) {
   for (std::size_t e = 0; e < plan.E; e++) {
      std::uint32_t i = plan.edges.i[e];
      std::uint32_t j = plan.edges.j[e];
      fn(pos, out, e, i, j);
   }
}

template <class Tag, typename T, class TensorT, class ParticlesT,
          class ParticlesView>
void pairwise_tag(const PairwiseMeta<T, ParticlesT> &meta,
                  ParticlesView &particles, TensorT &out,
                  params_type_t<Tag, T> params) {

   //      if (!meta.plan->topology.crs) {
   //    throw std::runtime_error("gather_index_tag: missing topology CRS");
   //}

   if (meta.fastpath) {
      pairwise_traits<Tag, T, ParticlesView>::can_execute(
          particles, meta.plan.topology.crs, out.get_ptr(),
          meta.plan.topology.E, params);
      return;
   }
   // return;
}

template <class Tag, typename T, class TensorT, class ParticlesT,
          class ParticlesView>
void gather_index_tag(const GatherIndexMeta<T, ParticlesT> &meta,
                      ParticlesView &particles, TensorT &out,
                      params_type_t<Tag, T> params) {

   if (meta.fastpath) {
      pairwise_traits<Tag, T, ParticlesView>::can_execute(
          particles, meta.plan.topology.crs, out.get_ptr(),
          meta.plan.topology.E, params);
      return;
   }
   // return;
}

} // namespace fusion::physics::iter

#endif // FUSION_CORE_ITER_TOPO_ITER_HPP
