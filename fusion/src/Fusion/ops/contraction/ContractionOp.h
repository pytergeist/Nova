#ifndef FUSION_OPS_CONTRACTION_OP_H
#define FUSION_OPS_CONTRACTION_OP_H

#include "Fusion/core/planning/OpContextBuilders.h"
#include "Fusion/execution/cpu/Contraction.h"
#include "Fusion/ops/OperandValidation.h"
#include "Fusion/ops/OutputAllocation.h"

namespace fusion::ops {

template <typename T, class OpTag, class ScalarTag>
DenseTensor<T> apply_contraction_op(const DenseTensor<T> &lhs,
                                    const DenseTensor<T> &rhs) {
   validation::validate_dense_contraction_operation<T, OpTag>(lhs, rhs);
   planning::ContractionContext ctx =
       planning::make_matmul_context<T>(lhs, rhs);
   DenseTensor<T> out = detail::init_contraction_out_from_ctx(lhs, rhs, ctx);
   execution::cpu::contraction<T, OpTag, ScalarTag>(
       out.get_ptr(), lhs.get_ptr(), rhs.get_ptr(), ctx);
   return out;
}
} // namespace fusion::ops

#endif // FUSION_OPS_CONTRACTION_OP_H