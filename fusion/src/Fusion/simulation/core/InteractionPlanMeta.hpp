// InteractionPlanMeta.hpp
// Umbrella header for physics dispatch/runtime metadata structs (PairwiseMeta,
// ...)

#ifndef FUSION_PHYSICS_META_HPP
#define FUSION_PHYSICS_META_HPP

#include <cstddef>

#include "Fusion/core/fuir/Descs.h"
#include "Neighbours.hpp"
#include "ParticleDescs.h"
#include "InteractionPlan.h"
#include "InteractionIR.h"

template <typename T, class ParticlesT> struct PairwiseMeta {
   bool fastpath;
   std::size_t fast_len;
   std::vector<std::size_t> out_shape;
   PairwisePlan plan;
};

template <typename T, class ParticlesT> struct GatherIndexMeta {
   bool fastpath;
   std::size_t fast_len;
   GatherIndexPlan plan;
};

template <typename T, class ParticlesT>
static OperandDescription
make_indexed_desc_from_particles_field(const ParticlesT &p) {
   OperandDescription d;
   d.shape = p.logical_shape();
   d.itemsize = sizeof(T);

   if constexpr (requires { p.x.strides(); }) {
      d.strides = p.logical_strides();
   } else {
      d.strides = contig_elem_strides(d.shape);
   }
   d.access = AccessKind::Indexed;
   d.layout = LayoutKind::AoSoA;
   d.storage = StorageKind::Owned;

   d.type = OperandDescType::Tensor;
   return d;
}

template <typename indexT = std::uint32_t>
OperandDescription
make_indexed_desc_from_topology_domain(const EdgeList &edges) {
   OperandDescription d;
   d.shape = {edges.E()};
   d.itemsize = sizeof(indexT);
   d.strides = {1};

   d.access = AccessKind::Indexed;
   d.layout = LayoutKind::Dense;
   d.storage = StorageKind::View;
   d.update = UpdateKind::ReadOnly;
   d.type = OperandDescType::Topology;
   return d;
}

template <typename T>
OperandDescription
make_indexed_desc_from_shape(const std::vector<std::size_t> &shape,
                             const int64_t *strides_elems) {
   OperandDescription d;
   std::vector<std::size_t> sz(shape.begin(), shape.end());
   std::vector<std::int64_t> st;
   if (strides_elems) {
      st.assign(strides_elems,
                strides_elems + static_cast<int64_t>(shape.size()));
   } else {
      st = contig_elem_strides(shape);
   }
   d.access = AccessKind::Indexed;
   d.layout = LayoutKind::AoSoA;
   d.storage = StorageKind::Owned;
   d.shape = std::move(sz);
   d.strides = std::move(st);
   d.itemsize = sizeof(T);
   return d;
}

template <typename T, class ParticlesT>
ParticlesAoSoADesc make_particles_aosoa_desc(const ParticlesT &particles,
                                             const EdgeList &edges) {
   ParticlesAoSoADesc desc;

   desc.N = particles.N();
   desc.E = edges.E();
   desc.tile = particles.tile();
   desc.dim = particles.dim();

   // TODO: this should make indexed the layout
   OperandDescription dX = make_desc_from_tensor(particles.x);
   OperandDescription dF = make_desc_from_tensor(particles.v);
   OperandDescription dV = make_desc_from_tensor(particles.f);

   desc.x_desc = dX;
   desc.f_desc = dF;
   desc.v_desc = dV;
   desc.itemsize = sizeof(T);
   return desc;
}

template <typename T, class ParticlesT>
PairwiseMeta<T, ParticlesT> make_pairwise_meta(const ParticlesT &psoa,
                                               EdgeList &edges,
                                               std::size_t channels) {
   PairwiseMeta<T, ParticlesT> meta;
   ParticlesAoSoADesc pdesc =
       make_particles_aosoa_desc<T, ParticlesT>(psoa, edges);
   meta.fast_len = edges.E();
   meta.plan = make_pairwise_plan(pdesc, edges);
   meta.out_shape = std::vector<std::size_t>{
       channels, static_cast<std::size_t>(meta.plan.E)};
   bool fcond = meta.plan.format == PairIndexFormat::PairBlockedCRS;
   bool lcond = meta.plan.layout == ParticleLayout::AoSoA;
   meta.fastpath = fcond && lcond;
   return meta;
}

template <typename T, class ParticlesT>
GatherIndexMeta<T, ParticlesT>
construct_gather_index_meta(const ParticlesT &particles, EdgeList &edges) {

   std::size_t channels = ParticlesT::dim();

   GatherIndexMeta<T, ParticlesT> meta;

   OperandDescription x_desc = make_indexed_desc_from_particles_field<T, ParticlesT>(particles);
   OperandDescription e_desc = make_indexed_desc_from_topology_domain<std::uint32_t>(edges);

   ParticlesAoSoADesc pdesc = make_particles_aosoa_desc<T, ParticlesT>(particles, edges);

   PairBlockedCRS crs = build_pair_index_blocked_crs_from_particle_field(pdesc, edges);

   const PairTopologyView ptv = make_pair_topology_view(edges, crs);
   const OperandLabelBinding binding = make_gather_index_label_binding(x_desc.ndims(), e_desc.ndims());
   const std::vector<OperandDescription> descs{x_desc, e_desc};
   meta.plan = make_gather_index_plan_from_binding_and_topo(descs, ptv, binding);
   meta.fast_len = edges.E();
   bool fcond = meta.plan.topology.format == PairIndexFormat::PairBlockedCRS;
   bool lcond = meta.plan.topology.layout == ParticleLayout::AoSoA;
   meta.fastpath = fcond && lcond;
   return meta;
}

#endif // FUSION_PHYSICS_META_HPP
