#ifndef OPS_REDUCE_H
#define OPS_REDUCE_H

#include <string_view>
#include <vector>

#include "Fusion/Tensor.h"
#include "Fusion/core/ElementWise.h"
#include "Fusion/core/Ffunc.h"
#include "Fusion/core/Reduce.h"
#include "Fusion/cpu/SimdTags.h"
#include "Fusion/cpu/SimdTraits.h"

#include "Helpers.h"

namespace math {

template <typename T> inline Tensor<T> sum(const Tensor<T> &x) { // TODO: This bypasses Tensor buffer/boadcast in curr impl
   const T *y = x.get_ptr();
   const std::size_t n = x.flat_size();
   T acc = reduce::reduce_tag<T, GlobalSumSIMD>(y, n);
   return Tensor<T>({1}, std::vector<T>{acc}, x.dtype(), Device::CPU, x.requires_grad());
}

template <typename T> inline Tensor<T> mean(const Tensor<T> &x) {
   const T *y = x.get_ptr();
   const std::size_t n = x.flat_size();
   T acc = reduce::reduce_tag<T, GlobalSumSIMD>(y, n);
   T mean = acc / static_cast<T>(n);
   return Tensor<T>({1}, std::vector<T>{acc}, x.dtype(), Device::CPU, x.requires_grad());
}
} // namespace math

#endif // OPS_REDUCE_H
