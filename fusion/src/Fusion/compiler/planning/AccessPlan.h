#ifndef FUSION_CORE_PLANNING_ACCESS_PLAN_H
#define FUSION_CORE_PLANNING_ACCESS_PLAN_H

#include <vector>

#include "Fusion/compiler/ir/IR.h"

namespace fusion::planning {
struct AccessPlan {
   std::vector<fuir::OperandAccess> operands{};
};
} // namespace fusion::planning
#endif // FUSION_CORE_PLANNING_ACCESS_PLAN_H