#ifndef FUSION_OPS_ALIGNED_COMPARISON_HPP
#define FUSION_OPS_ALIGNED_COMPARISON_HPP

#include <string_view>
#include <vector>

#include "Fusion/compiler/planning/OpContextBuilders.h"
#include "Fusion/execution/cpu/BinaryElementwise.h"
#include "Fusion/ops/OperandValidation.h"
#include "Fusion/ops/OutputAllocation.h"

namespace fusion::ops::aligned {

template <typename T>
DenseTensor<T> greater(const DenseTensor<T> &x, const DenseTensor<T> &y) {
   return apply_binary_op<T, GreaterTag>(x, y);
}

template <typename T>
DenseTensor<T> greater_equal(const DenseTensor<T> &x, const DenseTensor<T> &y) {
   return apply_binary_op<T, GreaterEqualTag>(x, y);
}

template <typename T>
DenseTensor<T> maximum(const DenseTensor<T> &x, const DenseTensor<T> &y) {
   return apply_binary_op<T, MaximumTag>(x, y);
}

} // namespace fusion::ops::aligned

#endif // FUSION_OPS_ALIGNED_COMPARISON_HPP
