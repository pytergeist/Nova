#ifndef FUSION_OPS_ALIGNED_UNARY_HPP
#define FUSION_OPS_ALIGNED_UNARY_HPP

#include "Fusion/core/planning/OpContextBuilders.h"
#include "Fusion/ops/OperandValidation.h"
#include "Fusion/ops/elementwise/UnaryOp.h"

namespace fusion::ops::aligned {

template <typename T> DenseTensor<T> sqrt(const DenseTensor<T> &x) {
   return apply_unary_op<T, SqrtTag>(x);
}

template <typename T> DenseTensor<T> log(const DenseTensor<T> &x) {
   return apply_unary_op<T, LogTag>(x);
}

template <typename T> DenseTensor<T> exp(const DenseTensor<T> &x) {
   return apply_unary_op<T, ExpTag>(x);
}

} // namespace fusion::ops::aligned

#endif // FUSION_OPS_ALIGNED_UNARY_HPP
