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
inline RawTensor<T>
global_sum(const RawTensor<T> &x) { // TODO: This bypasses Tensor
   // buffer/boadcast in curr impl
   const T *y = x.get_ptr();
   const std::size_t n = x.flat_size();
   T acc = reduce::reduce_tag<T, GlobalSumSIMD>(y, n);
   return RawTensor<T>({1}, std::vector<T>{acc}, x.dtype(), x.device());
}

template <typename T> inline RawTensor<T> global_mean(const RawTensor<T> &x) {
   const T *y = x.get_ptr();
   const std::size_t n = x.flat_size();
   T acc = reduce::reduce_tag<T, GlobalSumSIMD>(y, n);
   T mean = acc / static_cast<T>(n);
   return RawTensor<T>({1}, std::vector<T>{acc}, x.dtype(), x.device());
}

template <typename T>
inline RawTensor<T> sum(const RawTensor<T> &x, const std::size_t axis,
                        const bool keep_dim) {
   if ((axis == -1) && !(keep_dim)) {
      RawTensor<T> out = global_sum(x);
      return out;
   }
   ReductionMeta meta = make_reduction_meta(x, axis, keep_dim);
   RawTensor<T> out = init_out_from_meta(x, meta);
   reduction::reduction_tag<T, GlobalSumSIMD>(x, meta, out);
   return out;
}

template <typename T>
inline RawTensor<T> mean(const RawTensor<T> &x, const std::size_t axis,
                         const bool keep_dim) {
   if ((axis == -1) && !(keep_dim)) {
      RawTensor<T> out = global_mean(x);
      return out;
   } else {
      throw std::runtime_error("Not implemented");
   }
//   ReductionMeta meta = make_reduction_meta(x, axis, keep_dim);
//   RawTensor<T> out = init_out_from_meta(x, meta);
//   const std::size_t n = x.flat_size();
//   T mean = out / static_cast<T>(n);
//   reduction::reduction_tag<T, GlobalSumSIMD>(x, meta, out);
//   return out;
}

} // namespace math

} // namespace fusion

#endif // OPS_REDUCE_HPP
