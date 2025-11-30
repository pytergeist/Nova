#ifndef EWISE_H
#define EWISE_H

#include <string_view>
#include <vector>

#include "Fusion/core/TensorBase.h"
#include "Fusion/core/ElementWise.h"
#include "Fusion/core/EwiseMeta.h"
#include "Fusion/storage/StorageInterface.h"

#include "Helpers.h"

namespace fusion {

namespace math {

template <typename T>
inline TensorBase<T> add(const TensorBase<T> &x, const TensorBase<T> &y) {
   BinaryEwiseMeta meta = make_binary_meta(x, y);
   TensorBase<T> out = init_out_from_meta(x, y, meta);
   ewise::binary_ewise_tag<T, AddSIMD>(x, y, meta, out);
   return out;
}

template <typename T>
inline TensorBase<T> sub(const TensorBase<T> &x, const TensorBase<T> &y) {
   BinaryEwiseMeta meta = make_binary_meta(x, y);
   TensorBase<T> out = init_out_from_meta(x, y, meta);
   ewise::binary_ewise_tag<T, SubtractSIMD>(x, y, meta, out);
   return out;
}

template <typename T>
inline TensorBase<T> mul(const TensorBase<T> &x, const TensorBase<T> &y) {
   BinaryEwiseMeta meta = make_binary_meta(x, y);
   TensorBase<T> out = init_out_from_meta(x, y, meta);
   ewise::binary_ewise_tag<T, MultiplySIMD>(x, y, meta, out);
   return out;
}

template <typename T>
inline TensorBase<T> div(const TensorBase<T> &x, const TensorBase<T> &y) {
   BinaryEwiseMeta meta = make_binary_meta(x, y);
   TensorBase<T> out = init_out_from_meta(x, y, meta);
   ewise::binary_ewise_tag<T, DivideSIMD>(x, y, meta, out);
   return out;
}

template <typename T>
inline TensorBase<T> pow(const TensorBase<T> &x, const TensorBase<T> &y) {
   BinaryEwiseMeta meta = make_binary_meta(x, y);
   TensorBase<T> out = init_out_from_meta(x, y, meta);
   ewise::binary_ewise_tag<T, PowerSIMD>(x, y, meta, out);
   return out;
}

} // namespace math

} // namespace fusion

#endif // EWISE_H
