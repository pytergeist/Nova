#ifndef FUSION_CORE_PLANNING_PLAN_KINDS_H
#define FUSION_CORE_PLANNING_PLAN_KINDS_H

#include <cstdint>

namespace fusion::planning {
enum class ExprKind : std::uint8_t {
   Elementwise,
   Reduction,
   Contraction,
   Pairwise,
   IndexedMap,
};

enum class TraversalKind : std::uint8_t {
   Dense,
   Indexed,
};

enum class IndexedFormat : std::uint8_t {
   EdgeList,
   CRS,
   BlockedCRS,
};
} // namespace fusion::planning

#endif // FUSION_CORE_PLANNING_PLAN_KINDS_H