#ifndef FUSION_CORE_PLANNING_OP_PLANS_H
#define FUSION_CORE_PLANNING_OP_PLANS_H

#include <cstddef>
#include <string_view>

#include "Fusion/compiler/planning/ExecutionPlan.h"

namespace fusion::planning {
/// The execution plan for an ElementWise expression
struct ElementwisePlan {
   static constexpr std::string_view name = "Elementwise Plan";
   ExecutionPlan exec;
};

struct ReductionPlan {
   static constexpr std::string_view name = "Reduction Plan";
   ExecutionPlan exec;

   std::size_t reduction_axis{0};
   bool keep_dim{false};
};

struct ContractionPlan {
   static constexpr std::string_view name = "Contraction Plan";
   ExecutionPlan exec;
};

struct IndexedPlan {
   static constexpr std::string_view name = "Indexed Plan";
   ExecutionPlan exec;
};
} // namespace fusion::planning

#endif // FUSION_CORE_PLANNING_OP_PLANS_H