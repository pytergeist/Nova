// PhysicsPlan.h
// Umbrella header for physics execution plan structs (PairwisePlan, BondedPlan,
// ...)

#ifndef FUSION_PHYSICS_PLAN_HPP
#define FUSION_PHYSICS_PLAN_HPP

#include <cstddef>
#include <cstdint>

#include "Neighbours.hpp"
#include "State.hpp"

enum class PairIndexFormat { EdgeList, PairCRS, PairBlockedCRS };
enum class ParticleLayout { SoA, AoS, AoSoA };

struct PairwisePlan {
   PairIndexFormat format{PairIndexFormat::PairBlockedCRS};
   ParticleLayout layout{ParticleLayout::AoSoA};

   EdgeList edges;
   PairBlockedCRS crs;

   std::int64_t N{0};
   std::int64_t E{0};

   std::size_t itemsize;
};

template <typename T, class ParticlesT>
inline PairCRS build_pair_index_crs(const ParticlesT &psoa,
                                    const EdgeList &edges) {
   PairCRS crs;
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

struct ParticlesAoSoADesc {
   std::int64_t N{0};
   std::size_t E{0};
   std::size_t tile{0};
   std::size_t dim{0};

   OperandDescription x_desc;
   OperandDescription f_desc;
   OperandDescription v_desc;

   std::size_t itemsize{0};
};

PairBlockedCRS build_pair_index_blocked_crs(const ParticlesAoSoADesc &pdesc,
                                            EdgeList &edges);

PairwisePlan make_pairwise_plan(const ParticlesAoSoADesc &pdesc,
                                EdgeList &edges);

#endif // FUSION_PHYSICS_PLAN_HPP
