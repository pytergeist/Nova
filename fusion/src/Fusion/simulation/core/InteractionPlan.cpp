
#include "InteractionPlan.h"
#include "Fusion/core/fuir/IR.h"
#include "InteractionIR.h"

#include <Fusion/core/fuir/DescContraints.h>

PairBlockedCRS build_pair_index_blocked_crs_from_particle_field(
    const ParticlesAoSoADesc &pdesc, EdgeList &edges) {
   if (!(edges.sorted == SortType::Blockij)) {
      edges.sort_by_blocks(pdesc.tile);
   }
   PairBlockedCRS bcrs;
   bcrs.N = pdesc.N;
   bcrs.E = pdesc.E;
   bcrs.TILE = pdesc.tile; // TODO: curr just hardcoding for dev
   bcrs.nBlocks = (bcrs.N + bcrs.TILE - 1) / bcrs.TILE;

   bcrs.ib_ptr.assign(bcrs.nBlocks + 1, 0);
   std::vector<Group> groups;
   std::vector<std::uint32_t> group_counter;
   std::size_t prev_ib = 0;
   std::size_t prev_jb = 0;
   bool have_prev = false;
   for (std::size_t e = 0; e < bcrs.E; ++e) {
      const std::uint32_t src = edges.i[e];
      const std::uint32_t dst = edges.j[e];
      std::uint64_t ib = src / bcrs.TILE;
      std::uint64_t jb = dst / bcrs.TILE;
      if (have_prev && ib == prev_ib && jb == prev_jb) {
         groups.back().num_edges += 1;
      } else {
         groups.push_back(Group{.ib_idx=ib, .jb_idx=jb, .num_edges=1});
         bcrs.ib_ptr[ib + 1]++;
         bcrs.jb_idx.push_back(jb);

         prev_jb = jb;
         prev_ib = ib;
         have_prev = true;
      }
      std::uint16_t i_lane_idx = edges.i[e] % bcrs.TILE;
      std::uint16_t j_lane_idx = edges.j[e] % bcrs.TILE;
      bcrs.i_lane.push_back(i_lane_idx);
      bcrs.j_lane.push_back(j_lane_idx);
      bcrs.e_idx.push_back(e);
   }

   for (std::size_t k = 1; k < bcrs.ib_ptr.size(); ++k) {
      bcrs.ib_ptr[k] += bcrs.ib_ptr[k - 1];
   }

   bcrs.jb_ptr.assign(groups.size() + 1, 0);
   std::size_t psum = 0;
   for (std::size_t k = 1; k < bcrs.jb_ptr.size(); ++k) {
      psum += groups[k - 1].num_edges;
      bcrs.jb_ptr[k] += psum;
   }

   return bcrs;
}

PairwisePlan make_pairwise_plan(const ParticlesAoSoADesc &pdesc,
                                EdgeList &edges) {
   PairwisePlan plan;
   plan.format = PairIndexFormat::PairBlockedCRS;
   plan.layout = ParticleLayout::AoSoA; // TODO: cur defualting to AoSoA, should
   // have all options?

   PairBlockedCRS crs =
       build_pair_index_blocked_crs_from_particle_field(pdesc, edges);
   plan.crs = crs;
   plan.edges = edges;

   plan.N = static_cast<int64_t>(pdesc.N);
   plan.E = static_cast<int64_t>(edges.E());

   plan.itemsize = pdesc.itemsize;

   return plan;
}

GatherIndexPlan make_gather_index_plan_from_binding_and_topo_out(
    const std::vector<OperandDescription> &descs,
    const PairTopologyView &topology, const OperandLabelBinding &binding) {
   if (descs.size() != 3) {
      throw std::runtime_error("einsum_out: expected descs = {out, A, B}");
   }
   constexpr ItemSizeGroupConstraint constraint{
       ItemSizeGroupConstraint::TopologyAllowed};
   IndexSpaceIR ir = build_ir_from_label_binding(descs, binding, constraint);

   const std::vector<std::size_t> expected = out_shape_from_ir(ir);
   if (descs[0].shape != expected) {
      throw std::runtime_error(
          "einsum_out: out.shape does not match inferred out shape");
   }

   const std::vector<std::uint32_t> outer_order = ir.out_indices;

   const std::vector<std::uint32_t> &loop_order = ir.out_indices;

   GatherIndexPlan plan;
   plan.num_operands = descs.size();
   plan.itemsize = ir.itemsize;

   plan.topology = topology;

   plan.out_shape = descs[0].shape;

   const std::vector<IndexRole> role_of_id =
       compute_roles_for_gemm_like(ir, binding);
   plan.loop = lower_to_loops(ir, descs, loop_order, &role_of_id);
   plan.op_access = lower_operand_access(
       ir, descs, loop_order); // TODO: add role to operand access??

   return plan;
}

PairTopologyView make_pair_topology_view(const EdgeList &edges,
                                         const PairBlockedCRS &crs) {

   PairTopologyView topo;

   topo.edges = std::move(edges);
   topo.crs = std::move(crs);

   topo.format = PairIndexFormat::PairBlockedCRS;
   topo.layout = ParticleLayout::AoSoA;

   topo.N = static_cast<int64_t>(topo.crs.N);
   topo.E = static_cast<int64_t>(topo.edges.E());

   return topo;
}

GatherIndexPlan make_gather_index_plan_from_binding_and_topo(
    const std::vector<OperandDescription> &inputs,
    const PairTopologyView &topology, const OperandLabelBinding &binding) {
   if (inputs.size() != 2) {
      throw std::runtime_error("einsum: expected inputs = {A, B}");
   }

   OperandDescription dummy_out;
   dummy_out.shape.assign(binding.out_labels.size(), 1);
   dummy_out.strides.assign(dummy_out.ndims(), 0);
   dummy_out.itemsize = inputs[0].itemsize;

   ItemSizeGroupConstraint constraint{ItemSizeGroupConstraint::TopologyAllowed};

   std::vector<OperandDescription> tmp = {dummy_out, inputs[0], inputs[1]};
   IndexSpaceIR ir = build_ir_from_label_binding(tmp, binding, constraint);

   const std::vector<std::size_t> out_shape = out_shape_from_ir(ir);

   OperandDescription out_desc;
   out_desc.shape = out_shape;

   out_desc.strides.assign(out_desc.ndims(), 0);

   out_desc.itemsize = inputs[0].itemsize;

   std::vector<OperandDescription> const descs{out_desc, inputs[0], inputs[1]};
   return make_gather_index_plan_from_binding_and_topo_out(descs, topology,
                                                           binding);
}