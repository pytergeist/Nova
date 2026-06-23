#ifndef FUSION_CORE_TOPOLOGY_TOPOLOGY_VIEW
#define FUSION_CORE_TOPOLOGY_TOPOLOGY_VIEW

#include "PairIndex.hpp"

// TODO: this needs to be moved inside planning system
enum class PairIndexFormat { EdgeList, PairCRS, PairBlockedCRS };
enum class ParticleLayout { SoA, AoS, AoSoA };

struct BlockedTopologyView {
   EdgeList edges;
   BlockedCRS crs;
   PairIndexFormat format{PairIndexFormat::PairBlockedCRS};
   ParticleLayout layout{ParticleLayout::AoSoA};
   std::int64_t N{0};
   std::int64_t E{0};
};



#endif // FUSION_CORE_TOPOLOGY_TOPOLOGY_VIEW