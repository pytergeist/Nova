#ifndef FUSION_CORE_ITER_DENSE_ITER_PLAN_VIEW
#define FUSION_CORE_ITER_DENSE_ITER_PLAN_VIEW

#include "Fusion/core/planning/TensorPlan.h"

#include <cstddef>
#include <span>
#include <variant>

#include "Fusion/common/Checks.hpp"

namespace fusion::dense::iter {
struct DenseIterPlanView {
   std::size_t num_operands{};
   std::span<const LoopDim> loop;
   std::span<const OperandAccess> operands;
};

template <typename IterPlan>
DenseIterPlanView dense_iter_view(const IterPlan &plan) {
   const DenseTraversalPlan *dense =
       std::get_if<DenseTraversalPlan>(&plan.exec.traversal);
   FUSION_CHECK(dense != nullptr, "Expected DenseTraversalPlan, got nullptr");
   return DenseIterPlanView{.num_operands = plan.exec.core.num_operands,
                            .loop = dense->loop,
                            .operands = plan.exec.access.operands};
}

} // namespace fusion::dense::iter

#endif // FUSION_CORE_ITER_DENSE_ITER_PLAN_VIEW