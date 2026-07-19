#ifndef FUSION_OPS_OUTPUT_ALLOCATION_H
#define FUSION_OPS_OUTPUT_ALLOCATION_H

#include "Fusion/core/planning/OpContextBuilders.h"

#include "Fusion/core/planning/OpContext.h"

namespace fusion::ops::detail {
template <typename T>
DenseTensor<T> init_out_from_meta(const DenseTensor<T> &x,
                                  const DenseTensor<T> &y,
                                  const planning::BinaryEwiseContext &m) {
   return DenseTensor<T>(m.out_shape, x.dtype(), x.device());
}

template <typename T>
DenseTensor<T> init_out_from_meta(const DenseTensor<T> &x,
                                  const planning::UnaryEwiseContext &m) {
   return DenseTensor<T>(m.out_shape, x.dtype(), x.device());
}

template <typename T>
DenseTensor<T> init_out_from_meta(const DenseTensor<T> &x,
                                  const fusion::planning::ReductionContext &m) {
   return DenseTensor<T>(m.out_shape, x.dtype(), x.device());
}

template <typename T>
DenseTensor<T> init_out_from_meta(const DenseTensor<T> &x,
                                  const DenseTensor<T> &y,
                                  const planning::ContractionContext &m) {
   return DenseTensor<T>(m.out_shape, x.dtype(), x.device());
}
} // namespace fusion::ops::detail
#endif // FUSION_OPS_OUTPUT_ALLOCATION_H
