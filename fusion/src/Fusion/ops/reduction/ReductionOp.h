#ifndef FUSION_OPS_ELEMENTWISE_REDUCTION_OP_H
#define FUSION_OPS_ELEMENTWISE_REDUCTION_OP_H

#include "Fusion/core/planning/OpContextBuilders.h"
#include "Fusion/execution/cpu/Reduction.h"
#include "Fusion/ops/OperandValidation.h"
#include "Fusion/ops/OutputAllocation.h"

namespace fusion::ops {

template <typename T, class OpTag>
DenseTensor<T> apply_reduction_op(const DenseTensor<T> &operand,
                                  const std::size_t axis, const bool keep_dim) {
   validation::validate_dense_reduction_operation<T, OpTag>(operand);
   planning::ReductionContext ctx =
       planning::make_reduction_context(operand, axis, keep_dim);
   DenseTensor<T> out = detail::init_reduction_out_from_ctx(operand, ctx);
   execution::cpu::reduction<T, OpTag>(out.get_ptr(), operand.get_ptr(),
                                       out.flat_size(), ctx);
   return out;
}
} // namespace fusion::ops

#endif // FUSION_OPS_ELEMENTWISE_REDUCTION_OP_H