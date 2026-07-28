#ifndef FUSION_OPS_ELEMENTWISE_BINARY_OP_H
#define FUSION_OPS_ELEMENTWISE_BINARY_OP_H

#include "Fusion/compiler/planning/OpContextBuilders.h"
#include "Fusion/execution/cpu/BinaryElementwise.h"
#include "Fusion/ops/OperandValidation.h"
#include "Fusion/ops/OutputAllocation.h"

namespace fusion::ops {

template <typename T, class OpTag>
DenseTensor<T> apply_binary_op(const DenseTensor<T> &lhs,
                               const DenseTensor<T> &rhs) {
   validation::validate_dense_binary_operation<T, OpTag>(lhs, rhs);
   planning::BinaryEwiseContext ctx =
       planning::make_binary_ewise_context(lhs, rhs);
   DenseTensor<T> out = detail::init_out_from_meta(lhs, rhs, ctx);
   execution::cpu::binary_elementwise<T, OpTag>(out.get_ptr(), lhs.get_ptr(),
                                                rhs.get_ptr(), ctx);
   return out;
}
} // namespace fusion::ops

#endif // FUSION_OPS_ELEMENTWISE_BINARY_OP_H