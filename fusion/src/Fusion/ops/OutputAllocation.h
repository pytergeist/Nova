#ifndef FUSION_OPS_OUTPUT_ALLOCATION_H
#define FUSION_OPS_OUTPUT_ALLOCATION_H

#include "Fusion/compiler/planning/OpContextBuilders.h"
#include "Fusion/compiler/planning/OpContext.h"

namespace fusion::ops::detail {
template <typename T>
DenseTensor<T> init_out_from_meta(const DenseTensor<T> &x,
                                  const DenseTensor<T> &y, // TODO: hanging arg?
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
                                  const planning::ReductionContext &m) {
   return DenseTensor<T>(m.out_shape, x.dtype(), x.device());
}

template <typename T>
DenseTensor<T> init_out_from_meta(const DenseTensor<T> &x,
                                  const DenseTensor<T> &y, // TODO: hanging arg?
                                  const planning::ContractionContext &m) {
   return DenseTensor<T>(m.out_shape, x.dtype(), x.device());
}

template <typename T>
DenseTensor<T> init_reduction_out_from_ctx(const DenseTensor<T> &x,
                                           const planning::ReductionContext &m,
                                           const T identity = T{0}) {

   DenseTensor<T> out = init_out_from_meta(x, m);

   std::fill(out.get_ptr(), out.get_ptr() + out.flat_size(), identity);

   return out;
}

template <typename T>
DenseTensor<T>
init_contraction_out_from_ctx(const DenseTensor<T> &x, const DenseTensor<T> &y,
                              const planning::ContractionContext &ctx,
                              const T identity = T{0}) {

   DenseTensor<T> out = init_out_from_meta(x, y, ctx);

   std::fill(out.get_ptr(), out.get_ptr() + out.flat_size(), identity);

   return out;
}

} // namespace fusion::ops::detail
#endif // FUSION_OPS_OUTPUT_ALLOCATION_H
