#ifndef FUSION_OPS_REDUCTION_REDUCE_HPP
#define FUSION_OPS_REDUCTION_REDUCE_HPP

#include "ReductionOp.h"

namespace fusion::ops::reduction {

template <typename T>
DenseTensor<T> sum(const DenseTensor<T> &x, const std::size_t axis,
                   const bool keep_dim) {
   return apply_reduction_op<T, SumTag>(x, axis, keep_dim);
}

template <typename T>
DenseTensor<T> mean(const DenseTensor<T> &x, const std::size_t axis,
                    const bool keep_dim) {
   return apply_reduction_op<T, MeanTag>(x, axis, keep_dim);
}

} // namespace fusion::ops::reduction

#endif // FUSION_OPS_REDUCTION_REDUCE_HPP
