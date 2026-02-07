#ifndef FUSION_PHYSICS_PLAN_HPP
#define FUSION_PHYSICS_PLAN_HPP

#include <cstdint>
#include <cstddef>

#include "State.hpp"

enum class PairListKind { EdgeList, CRS };

enum class VecLayout { SoA, AoS };

struct PairwisePlan {
   PairListKind kind{ PairListKind::EdgeList };
   VecLayout layout{ VecLayout::SoA };

   std::int64_t N{0};
   std::int64_t E{0};
   bool x_contig{false};
   bool f_contig{false};

   std::size_t itemsize;

};

template<typename T>
inline PairwisePlan make_pairwise_plan(const ParticlesSoA<T>& psoa,
                                       const EdgeList& edges) {
   PairwisePlan plan;
   plan.kind = PairListKind::EdgeList;
   plan.layout = VecLayout::SoA;

   plan.N = static_cast<int64_t>(psoa.N());
   plan.E = static_cast<int64_t>(edges.size());
   plan.x_contig = psoa.x.is_contiguous();
   plan.f_contig = psoa.f.is_contiguous();
   plan.itemsize = sizeof(T);

   return plan;
}

#endif // FUSION_PHYSICS_PLAN_HPP
