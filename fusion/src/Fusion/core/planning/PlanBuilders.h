#ifndef FUSION_CORE_PLANNING_PLAN_BUILDERS_H
#define FUSION_CORE_PLANNING_PLAN_BUILDERS_H

#include <cstddef>
#include <vector>

#include "Fusion/core/fuir/IR.h"
#include "OpPlans.h"

namespace fusion::planning {
ElementWisePlan
make_elementwise_plan(const std::vector<OperandDescription> &descs);

ReductionPlan make_reduction_plan(const std::vector<OperandDescription> &desc,
                                  const std::size_t axis, const bool keepdim);

ContractionPlan
make_contraction_plan_einsum(const std::vector<OperandDescription> &inputs,
                             const OperandLabelBinding &binding);

ContractionPlan
make_contraction_plan_einsum_out(const std::vector<OperandDescription> &descs,
                                 const OperandLabelBinding &binding);

IndexedPlan make_indexed_plan(const std::vector<OperandDescription> &descs,
                              const BlockedCRS &bcrs, const EdgeList &edges);
} // namespace fusion::planning

#endif // FUSION_CORE_PLANNING_PLAN_BUILDERS_H