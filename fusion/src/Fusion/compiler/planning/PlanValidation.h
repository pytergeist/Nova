#ifndef FUSION_CORE_PLANNING_PLAN_VALIDATION_H
#define FUSION_CORE_PLANNING_PLAN_VALIDATION_H

#include <string_view>

#include "Fusion/compiler/ir/IR.h"
#include "Fusion/compiler/planning/ExecutionPlan.h"

namespace fusion::planning::validation {

void validate_operand_description(const fuir::OperandDescription &desc,
                                  std::string_view where);

void validate_execution_plan(const ExecutionPlan &plan, std::string_view where);

void validate_dense_execution_plan(const ExecutionPlan &plan,
                                   std::string_view where);

void validate_indexed_execution_plan(const ExecutionPlan &plan,
                                     std::string_view where);

} // namespace fusion::planning::validation

#endif // FUSION_CORE_PLANNING_PLAN_VALIDATION_H