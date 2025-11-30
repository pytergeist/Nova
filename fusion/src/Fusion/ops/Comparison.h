#ifndef OPS_COMPARISON_H
#define OPS_COMPARISON_H

#include <string_view>
#include <vector>

#include "Fusion/core/TensorBase.h"
#include "Fusion/core/Device.h"
#include "Fusion/core/ElementWise.h"
#include "Fusion/core/EwiseMeta.h"

#include "Helpers.h"

namespace fusion {

namespace math {

template <typename T>
inline TensorBase<T> greater(const TensorBase<T> &x, const TensorBase<T> &y) {
   BinaryEwiseMeta meta = make_binary_meta(x, y);
   TensorBase<T> out = init_out_from_meta(x, y, meta);
   ewise::binary_ewise_tag<T, GreaterThanSIMD>(x, y, meta, out);
   return out;
}

template <typename T>
inline TensorBase<T> greater_equal(const TensorBase<T> &x, const TensorBase<T> &y) {
   BinaryEwiseMeta meta = make_binary_meta(x, y);
   TensorBase<T> out = init_out_from_meta(x, y, meta);
   ewise::binary_ewise_tag<T, GreaterThanEqualSIMD>(x, y, meta, out);
   return out;
}

template <typename T>
inline TensorBase<T> maximum(const TensorBase<T> &x, const TensorBase<T> &y) {
   BinaryEwiseMeta meta = make_binary_meta(x, y);
   TensorBase<T> out = init_out_from_meta(x, y, meta);
   ewise::binary_ewise_tag<T, MaximumSIMD>(x, y, meta, out);
   return out;
}

} // namespace math

} // namespace fusion

#endif // OPS_COMPARISON_H
