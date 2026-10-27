#ifndef FUSION_CORE_PLANNING_ACCESS_PLAN_H
#define FUSION_CORE_PLANNING_ACCESS_PLAN_H

#include <vector>

#include "Fusion/compiler/lowering/Lowering.h"

namespace fusion::planning {
struct AccessPlan {
   std::vector<lowering::OperandAccess> operands{};
};
} // namespace fusion::planning
#endif // FUSION_CORE_PLANNING_ACCESS_PLAN_H