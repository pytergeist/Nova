#ifndef FUSION_OPS_ALIGNED_COMPARISON_HPP
#define FUSION_OPS_ALIGNED_COMPARISON_HPP

#include <string_view>
#include <vector>

#include "Fusion/core/planning/OpContextBuilders.h"
#include "Fusion/execution/cpu/BinaryElementwise.h"
#include "Fusion/ops/OperandValidation.h"
#include "Fusion/ops/OutputAllocation.h"

namespace fusion::ops::aligned {

template <typename T>
DenseTensor<T> greater(const DenseTensor<T> &x, const DenseTensor<T> &y) {
   validation::validate_dense_binary_operation<T, GreaterTag>(x, y);
   planning::BinaryEwiseContext meta =
       planning::make_binary_ewise_context(x, y);
   DenseTensor<T> out = detail::init_out_from_meta(x, y, meta);
   execution::cpu::binary_elementwise<T, GreaterThanSIMD>(x, y, meta, out);
   return out;
}

template <typename T>
DenseTensor<T> greater_equal(const DenseTensor<T> &x, const DenseTensor<T> &y) {
   validation::validate_dense_binary_operation<T, GreaterEqualTag>(x, y);
   planning::BinaryEwiseContext meta =
       planning::make_binary_ewise_context(x, y);
   DenseTensor<T> out = detail::init_out_from_meta(x, y, meta);
   execution::cpu::binary_elementwise<T, GreaterThanEqualSIMD>(x, y, meta, out);
   return out;
}

template <typename T>
DenseTensor<T> maximum(const DenseTensor<T> &x, const DenseTensor<T> &y) {
   validation::validate_dense_binary_operation<T, MaximumTag>(x, y);
   planning::BinaryEwiseContext meta =
       planning::make_binary_ewise_context(x, y);
   DenseTensor<T> out = detail::init_out_from_meta(x, y, meta);
   execution::cpu::binary_elementwise<T, MaximumSIMD>(x, y, meta, out);
   return out;
}

} // namespace fusion::ops::aligned

#endif // FUSION_OPS_ALIGNED_COMPARISON_HPP
