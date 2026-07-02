#ifndef FUSION_CORE_PLANNING_ACCESS_PLAN_H
#define FUSION_CORE_PLANNING_ACCESS_PLAN_H

#include <vector>

#include "Fusion/core/fuir/IR.h"

struct AccessPlan {
   std::vector<OperandAccess> operands;
};

#endif // FUSION_CORE_PLANNING_ACCESS_PLAN_H