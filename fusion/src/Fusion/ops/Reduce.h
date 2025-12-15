#ifndef OPS_REDUCE_H
#define OPS_REDUCE_H

#include <string_view>
#include <vector>

#include "Fusion/core/ElementWise.h"
#include "Fusion/core/Ffunc.h"
#include "Fusion/core/Reduce.h"
#include "Fusion/core/TensorBase.h"
#include "Fusion/cpu/SimdTags.hpp"
#include "Fusion/cpu/SimdTraits.hpp"

#include "Helpers.h"

namespace fusion {

namespace math {

template <typename T>
inline TensorBase<T> sum(const TensorBase<T> &x) { // TODO: This bypasses Tensor
   // buffer/boadcast in curr impl
   const T *y = x.get_ptr();
   const std::size_t n = x.flat_size();
   T acc = reduce::reduce_tag<T, GlobalSumSIMD>(y, n);
   return TensorBase<T>({1}, std::vector<T>{acc}, x.dtype(), x.device());
}

template <typename T> inline TensorBase<T> mean(const TensorBase<T> &x) {
   const T *y = x.get_ptr();
   const std::size_t n = x.flat_size();
   T acc = reduce::reduce_tag<T, GlobalSumSIMD>(y, n);
   T mean = acc / static_cast<T>(n);
   return TensorBase<T>({1}, std::vector<T>{acc}, x.dtype(), x.device());
}

} // namespace math

} // namespace fusion

#endif // OPS_REDUCE_H
