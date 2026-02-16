#ifndef FUSION_PHYSICS_PLAN_HPP
#define FUSION_PHYSICS_PLAN_HPP

#include <cstddef>
#include <cstdint>

#include "Neighbours.hpp"
#include "State.hpp"

enum class PairListKind { EdgeList, CRS, NMCLuster };

enum class VecLayout { SoA, AoS, AoSoA };

template <typename T, class ParticlesT> struct PairwisePlan {
   PairListKind kind{PairListKind::CRS};
   VecLayout layout{VecLayout::SoA};

   ParticlesT psoa;
   EdgeList edges;
   CRS crs;

   std::int64_t N{0};
   std::int64_t E{0};
   bool x_contig{false};
   bool f_contig{false};

   std::size_t itemsize;
};

template <typename T, class ParticlesT>
inline CRS make_crs(const ParticlesT &psoa, const EdgeList &edges) {
   CRS crs;
   crs.N = psoa.N();
   crs.E = edges.E();

   crs.row_ptr.assign(crs.N + 1, 0);
   crs.col_idx.resize(crs.E);

   for (std::size_t e = 0; e < crs.E; ++e) {
      const std::uint32_t src = edges.i[e];
      crs.row_ptr[src + 1]++;
   }

   for (std::int64_t i = 0; i < crs.N; ++i) {
      crs.row_ptr[i + 1] += crs.row_ptr[i];
   }

   std::vector<std::uint32_t> cursor = crs.row_ptr;
   for (std::size_t e = 0; e < crs.E; ++e) {
      const std::uint32_t src = edges.i[e];
      const std::uint32_t dst = edges.j[e];
      const std::uint32_t pos = cursor[src]++;
      crs.col_idx[pos] = dst;
   }
   return crs;
}

struct Group {
   std::uint64_t ib_idx;
   std::uint64_t jb_idx;
   std::uint64_t num_edges;
   bool operator==(const Group &g) const {
      return ib_idx == g.ib_idx && jb_idx == g.jb_idx;
   }
};

template <typename T, class ParticlesT>
inline BlockedCRS make_blocked_crs(const ParticlesT &psoa, EdgeList &edges) {
   if (!(edges.sorted == SortType::Blockij)) {
      edges.sort_by_blocks(psoa.tile());
   }
   BlockedCRS bcrs;
   bcrs.N = psoa.N();
   bcrs.E = edges.E();
   bcrs.TILE = psoa.tile(); // TODO: curr just hardcoding for dev
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
      bcrs.e_idx.push_back(edges.i[e]);
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

   std::cout << "Groups: [";
   for (auto v : groups) {
      std::cout << v.num_edges << ", ";
   }
   std::cout << "]" << std::endl;

   std::cout << "ib_ptr: [";
   for (auto i : bcrs.ib_ptr) {
      std::cout << i << ", ";
   }
   std::cout << "]" << std::endl;
   std::cout << "jb_idx: [";
   for (auto i : bcrs.jb_idx) {
      std::cout << i << ", ";
   }
   std::cout << "]" << std::endl;

   std::cout << "jb_ptr: [";
   for (auto i : bcrs.jb_ptr) {
      std::cout << i << ", ";
   }
   std::cout << "]" << std::endl;

   std::cout << "i_lane: [";
   for (auto i : bcrs.i_lane) {
      std::cout << i << ", ";
   }
   std::cout << "]" << std::endl;

   std::cout << "j_lane: [";
   for (auto i : bcrs.j_lane) {
      std::cout << i << ", ";
   }
   std::cout << "]" << std::endl;

   std::cout << "j_lane: [";
   for (auto i : bcrs.e_idx) {
      std::cout << i << ", ";
   }
   std::cout << "]" << std::endl;

   return bcrs;
}

template <typename T, class ParticlesT>
inline PairwisePlan<T, ParticlesT> make_pairwise_plan(const ParticlesT &psoa,
                                                      const EdgeList &edges) {
   PairwisePlan<T, ParticlesT> plan;
   plan.kind = PairListKind::CRS;
   plan.layout = VecLayout::AoSoA; // TODO: cur defualting to AoSoA, should have
                                   // all options?
   plan.psoa = psoa;

   CRS crs = make_crs<T, ParticlesT>(psoa, edges);
   plan.crs = crs;
   plan.edges = edges;

   plan.N = static_cast<int64_t>(psoa.N());
   plan.E = static_cast<int64_t>(edges.E());
   plan.x_contig = psoa.x.is_contiguous();
   plan.f_contig = psoa.f.is_contiguous();
   plan.itemsize = sizeof(T);

   return plan;
}

#endif // FUSION_PHYSICS_PLAN_HPP
