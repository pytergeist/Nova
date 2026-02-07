#ifndef FUSION_PHYSICS_PLAN
#define FUSION_PHYSICS_PLAN

#include <cstdint>

enum class PairListKind { EdgeList, CRS }

enum class VecLayout { SoA, AoS }

struct PairwisePlan {
   PairListKind kind{ PairlistKind::EdgeList };
   VecLayout layout{ VecLayout::SoA };

   std::int64_t N{0};
   std::int64_t E{0};
   bool x_contig{false};
   bool f_contig{false};

   std::size_t itemsize;

}

template<typename T>
PairwisePlan make_pairwise_plan(const ParticlesSoA<T>& psoa, const EdgeList& edges);

#endif // FUSION_PHYSICS_PLAN
