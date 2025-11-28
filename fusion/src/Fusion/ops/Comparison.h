#ifndef OPS_COMPARISON_H
#define OPS_COMPARISON_H

#include <string_view>
#include <vector>

#include "Fusion/Tensor.h"
#include "Fusion/core/ElementWise.h"
#include "Fusion/core/Device.h"
#include "Fusion/core/EwiseMeta.h"

#include "Helpers.h"


namespace math {
template <typename T>
inline Tensor<T> greater(const Tensor<T> &x, const Tensor<T> &y) {
   BinaryEwiseMeta meta = make_binary_meta(x, y);
   Tensor<T> out = init_out_from_meta(x, y, meta);
   ewise::binary_ewise_tag<T, GreaterThanSIMD>(x, y, meta, out);
   return out;
}

template <typename T>
inline Tensor<T> greater_equal(const Tensor<T> &x, const Tensor<T> &y) {
   BinaryEwiseMeta meta = make_binary_meta(x, y);
   Tensor<T> out = init_out_from_meta(x, y, meta);
   ewise::binary_ewise_tag<T, GreaterThanEqualSIMD>(x, y, meta, out);
   return out;
}

template <typename T>
inline Tensor<T> maximum(const Tensor<T> &x, const Tensor<T> &y) {
   BinaryEwiseMeta meta = make_binary_meta(x, y);
   Tensor<T> out = init_out_from_meta(x, y, meta);
   ewise::binary_ewise_tag<T, MaximumSIMD>(x, y, meta, out);
   return out;
}
} // namespace math

#endif // OPS_COMPARISON_H
