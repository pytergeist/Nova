#ifndef FUSION_OPS_ELEMENTWISE_UNARY_OP_H
#define FUSION_OPS_ELEMENTWISE_UNARY_OP_H

#include "Fusion/core/planning/OpContextBuilders.h"
#include "Fusion/execution/cpu/UnaryElementwise.h"
#include "Fusion/ops/OperandValidation.h"
#include "Fusion/ops/OutputAllocation.h"

namespace fusion::ops {

template <typename T, class OpTag>
DenseTensor<T> apply_unary_op(const DenseTensor<T> &operand) {
   validation::validate_dense_unary_operation<T, OpTag>(operand);
   planning::UnaryEwiseContext meta =
       planning::make_unary_ewise_context(operand);
   DenseTensor<T> out = detail::init_out_from_meta(operand, meta);
   execution::cpu::unary_elementwise<T, OpTag>(out.get_ptr(), operand.get_ptr(),
                                               meta);
   return out;
}
} // namespace fusion::ops

#endif // FUSION_OPS_ELEMENTWISE_UNARY_OP_H