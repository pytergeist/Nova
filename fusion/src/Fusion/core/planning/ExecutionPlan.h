#ifndef FUSION_CORE_PLANNING_EXECUTION_PLAN_H
#define FUSION_CORE_PLANNING_EXECUTION_PLAN_H

#include <vector>
#include <string_view>

#include "KernelHints.h"
#include "PlanKinds.h"
#include "AccessPlan.h"
#include "TraversalPlan.h"

struct PlanCore {
   static constexpr std::string_view name = "Plan Core";
   ExprKind expr;
   TraversalKind traversal;

   std::size_t itemsize{0};
   std::size_t num_operands{0};
   std::size_t out_ndim{0};
   std::vector<std::size_t> out_shape;
};

struct ExecutionPlan {
   static constexpr std::string_view name = "Execution Plan";
   PlanCore core;
   TraversalPlan traversal;
   AccessPlan access;
   KernelHints hints;
};

#endif // FUSION_CORE_PLANNING_EXECUTION_PLAN_H