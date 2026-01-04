#ifndef OPS_REDUCE_HPP
#define OPS_REDUCE_HPP

#include <string_view>
#include <vector>

#include "Fusion/core/EwiseIter.hpp"
#include "Fusion/core/RawTensor.hpp"
#include "Fusion/core/Reduce.hpp"
#include "Fusion/core/ReductionIter.hpp"
#include "Fusion/cpu/SimdTags.hpp"
#include "Fusion/cpu/SimdTraits.hpp"

#include "Helpers.hpp"

namespace fusion {

namespace math {

template <typename T>
inline RawTensor<T> sum(const RawTensor<T> &x, const std::size_t axis,
                        const bool keep_dim) {
   ReductionMeta meta = make_reduction_meta(x, axis, keep_dim);
   RawTensor<T> out = init_out_from_meta(x, meta);
   reduction::reduction_tag<T, SumSIMD>(x, meta, out);
   return out;
}

template <typename T>
inline RawTensor<T> mean(const RawTensor<T> &x, const std::size_t axis,
                         const bool keep_dim) {
   ReductionMeta meta = make_reduction_meta(x, axis, keep_dim);
   RawTensor<T> out = init_out_from_meta(x, meta);
   reduction::reduction_tag<T, MeanSIMD>(x, meta, out);
   return out;
}

} // namespace math

} // namespace fusion

#endif // OPS_REDUCE_HPP
