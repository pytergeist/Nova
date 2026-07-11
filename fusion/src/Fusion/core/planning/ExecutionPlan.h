#ifndef FUSION_CORE_PLANNING_EXECUTION_PLAN_H
#define FUSION_CORE_PLANNING_EXECUTION_PLAN_H

#include <cstddef>
#include <string_view>
#include <vector>

#include "Fusion/core/planning/AccessPlan.h"
#include "Fusion/core/planning/KernelHints.h"
#include "Fusion/core/planning/PlanKinds.h"
#include "Fusion/core/planning/TraversalPlan.h"

namespace fusion::planning {
struct PlanCore {
   static constexpr std::string_view name = "Plan Core";
   ExprKind expr{};
   TraversalKind traversal_kind{};

   std::size_t itemsize{0};
   std::size_t num_operands{0};
   std::size_t out_ndim{0};
   std::vector<std::size_t> out_shape{};
};

struct ExecutionPlan {
   static constexpr std::string_view name = "Execution Plan";
   PlanCore core;
   TraversalPlan traversal;
   AccessPlan access;
   KernelHints hints;
};
} // namespace fusion::planning

#endif // FUSION_CORE_PLANNING_EXECUTION_PLAN_H