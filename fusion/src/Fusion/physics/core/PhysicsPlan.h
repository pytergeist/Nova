#ifndef FUSION_PHYSICS_PLAN_HPP
#define FUSION_PHYSICS_PLAN_HPP

#include <cstddef>
#include <cstdint>

#include "State.hpp"

enum class PairListKind { EdgeList, CRS, NMCLuster };

enum class VecLayout { SoA, AoS };

template <typename T> struct PairwisePlan {
   PairListKind kind{PairListKind::EdgeList};
   VecLayout layout{VecLayout::SoA};

   ParticlesSoA<T> psoa;
   EdgeList edges;

   std::int64_t N{0};
   std::int64_t E{0};
   bool x_contig{false};
   bool f_contig{false};

   std::size_t itemsize;
};

template <typename T>
inline PairwisePlan<T> make_pairwise_plan(const ParticlesSoA<T> &psoa,
                                          const EdgeList &edges) {
   PairwisePlan<T> plan;
   plan.kind = PairListKind::EdgeList;
   plan.layout = VecLayout::SoA;
   plan.psoa = psoa;
   plan.edges = edges;

   plan.N = static_cast<int64_t>(psoa.N());
   plan.E = static_cast<int64_t>(edges.size());
   plan.x_contig = psoa.x.is_contiguous();
   plan.f_contig = psoa.f.is_contiguous();
   plan.itemsize = sizeof(T);

   return plan;
}

#endif // FUSION_PHYSICS_PLAN_HPP
