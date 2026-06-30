#ifndef FUSION_OPS_REDUCTION_REDUCE_HPP
#define FUSION_OPS_REDUCTION_REDUCE_HPP

#include "Fusion/core/iter/TensorIter.hpp"
#include "Fusion/cpu/simd/SimdTags.hpp"

#include "../Helpers.hpp"

namespace fusion::ops::reduction {

template <typename T>
DenseTensor<T> sum(const DenseTensor<T> &x, const std::size_t axis,
                   const bool keep_dim) {
   require_reduction_out_of_place<SumTag>();
   ReductionMeta meta = make_reduction_meta(x, axis, keep_dim);
   DenseTensor<T> out = init_out_from_meta(x, meta);
   fusion::dense::iter::reduction_tag<T, SumSIMD>(x, meta, out);
   return out;
}

template <typename T>
DenseTensor<T> mean(const DenseTensor<T> &x, const std::size_t axis,
                    const bool keep_dim) {
   require_reduction_out_of_place<MeanTag>();
   ReductionMeta meta = make_reduction_meta(x, axis, keep_dim);
   DenseTensor<T> out = init_out_from_meta(x, meta);
   fusion::dense::iter::reduction_tag<T, SumSIMD>(x, meta, out);
   const T denom = static_cast<T>(meta.reduce_len);
   out = out / denom;
   return out;
}

} // namespace fusion::ops::reduction

#endif // FUSION_OPS_REDUCTION_REDUCE_HPP
