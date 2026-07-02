#ifndef FUSION_CORE_PLANNING_TRAVERSAL_PLAN_H
#define FUSION_CORE_PLANNING_TRAVERSAL_PLAN_H

#include <variant>
#include <vector>

#include "Fusion/core/fuir/IR.h"
#include "Fusion/core/topology/PairIndex.hpp"
#include "PlanKinds.h"

struct DenseTraversalPlan {
   std::vector<LoopDim> loop;
};

struct IndexedTraversalPlan {
   IndexedFormat format{IndexedFormat::BlockedCRS};
   BlockedCRS blocked_crs;
};

using TraversalPlan = std::variant<DenseTraversalPlan, IndexedTraversalPlan>;

#endif // FUSION_CORE_PLANNING_TRAVERSAL_PLAN_H