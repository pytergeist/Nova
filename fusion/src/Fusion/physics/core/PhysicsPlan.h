#ifndef FUSION_PHYSICS_PLAN_HPP
#define FUSION_PHYSICS_PLAN_HPP

#include <cstddef>
#include <cstdint>

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
