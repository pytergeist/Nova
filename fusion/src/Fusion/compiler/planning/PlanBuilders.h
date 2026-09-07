#ifndef FUSION_CORE_PLANNING_PLAN_BUILDERS_H
#define FUSION_CORE_PLANNING_PLAN_BUILDERS_H

#include <cstddef>
#include <vector>

#include "Fusion/compiler/ir/IR.h"
#include "Fusion/compiler/planning/OpPlans.h"

namespace fusion::planning {
ElementwisePlan
make_elementwise_plan(const std::vector<fuir::OperandDescription> &descs);

ReductionPlan
make_reduction_plan(const std::vector<fuir::OperandDescription> &desc,
                    const std::size_t axis, const bool keepdim);

ContractionPlan make_contraction_plan_from_binding(
    const std::vector<fuir::OperandDescription> &descs,
    const fuir::OperandLabelBinding &binding);

IndexedPlan
make_indexed_plan(const std::vector<fuir::OperandDescription> &descs,
                  const BlockedCRS &bcrs, const EdgeList &edges);
} // namespace fusion::planning

#endif // FUSION_CORE_PLANNING_PLAN_BUILDERS_H