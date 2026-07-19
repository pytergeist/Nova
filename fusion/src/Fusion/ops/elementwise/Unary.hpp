#ifndef FUSION_OPS_ALIGNED_UNARY_HPP
#define FUSION_OPS_ALIGNED_UNARY_HPP

#include <string_view>
#include <vector>

#include "Fusion/core/planning/OpContextBuilders.h"
#include "Fusion/execution/cpu/UnaryElementwise.h"
#include "Fusion/ops/OperandValidation.h"
#include "Fusion/ops/OutputAllocation.h"

namespace fusion::ops::aligned {

template <typename T> DenseTensor<T> sqrt(const DenseTensor<T> &x) {
   validation::validate_dense_unary_operation<T, SqrtTag>(x);
   planning::UnaryEwiseContext meta = planning::make_unary_ewise_context(x);
   DenseTensor<T> out = detail::init_out_from_meta(x, meta);
   execution::cpu::unary_elementwise<T, SqrtSIMD>(x, meta, out);
   return out;
}

template <typename T> DenseTensor<T> log(const DenseTensor<T> &x) {
   validation::validate_dense_unary_operation<T, LogTag>(x);
   planning::UnaryEwiseContext meta = planning::make_unary_ewise_context(x);
   DenseTensor<T> out = detail::init_out_from_meta(x, meta);
   execution::cpu::unary_elementwise<T, NaturalLogSIMD>(x, meta, out);
   return out;
}

template <typename T> DenseTensor<T> exp(const DenseTensor<T> &x) {
   validation::validate_dense_unary_operation<T, ExpTag>(x);
   planning::UnaryEwiseContext meta = planning::make_unary_ewise_context(x);
   DenseTensor<T> out = detail::init_out_from_meta(x, meta);
   execution::cpu::unary_elementwise<T, ExponentialSIMD>(x, meta, out);
   return out;
}

} // namespace fusion::ops::aligned

#endif // FUSION_OPS_ALIGNED_UNARY_HPP
