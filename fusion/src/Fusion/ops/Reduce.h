#ifndef OPS_REDUCE_H
#define OPS_REDUCE_H

#include "../Tensor.h"
#include "../core/ElementWise.h"
#include "../core/Ffunc.h"
#include "../core/Reduce.h"
#include "../cpu/SimdTags.h"
#include "../cpu/SimdTraits.h"
#include "Helpers.h"
#include <string_view>
#include <vector>

namespace math {

template <typename T> inline Tensor<T> sum(const Tensor<T> &x) {
   const T *y = x.storage->data_ptr();
   const std::size_t n = x.flat_size();
   T acc = reduce::reduce_tag<T, GlobalSumSIMD>(y, n);
   return Tensor<T>({1}, std::vector<T>{acc}, Device::CPU, x.requires_grad());
}

template <typename T> inline Tensor<T> mean(const Tensor<T> &x) {
   const T *y = x.storage->data_ptr();
   const std::size_t n = x.flat_size();
   T acc = reduce::reduce_tag<T, GlobalSumSIMD>(y, n);
   T mean = acc / static_cast<T>(n);
   return Tensor<T>({1}, std::vector<T>{acc}, Device::CPU, x.requires_grad());
}
} // namespace math

#endif // OPS_REDUCE_H
