
#include "PhysicsPlan.h"
#include "PhysicsIR.h"

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
         groups.push_back(Group{ib, jb, 1});
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

GatherIndexPlan
make_gather_index_plan_with_blocked_crs(const std::vector<OperandDescription> &descs,
                                        PairBlockedCRS &bcrs, // TODO: eval const qualifier
                                        EdgeList &edges) {
   GatherIndexPlan plan;
   plan.N = static_cast<int64_t>(bcrs.N);
   plan.E = static_cast<int64_t>(edges.E());
   plan.format =
       PairIndexFormat::PairBlockedCRS; // TODO: This curr doesn't exist in fuir
   plan.layout = ParticleLayout::AoSoA;

   IndexSpaceIR ir = build_gather_and_map_ir(descs);

   plan.crs = bcrs; // TODO: do we want to move here, who owns bcrs?
   plan.edges = edges;

   const std::vector<std::uint32_t> &loop_order = ir.out_indices;

   plan.loop = lower_to_loops(ir, descs, loop_order);
   plan.op_access = lower_operand_access(ir, descs, loop_order);


   plan.itemsize = descs[0].itemsize;

   return plan;
}
