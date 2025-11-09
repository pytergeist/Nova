#ifndef EWISE_H
#define EWISE_H

#include <string_view>
#include <vector>
#include "../Tensor.h"
#include "../core/ElementWise.h"
#include "Helpers.h"
#include "../storage/StorageInterface.h"
#include "../core/EwiseMeta.h"

namespace math {

template <typename T>
inline Tensor<T> add(const Tensor<T> &x, const Tensor<T> &y) {
    BinaryEwiseMeta meta = make_binary_meta(x, y);
    Tensor<T> out = init_out_from_meta(x, y, meta);
    ewise::binary_ewise_tag<T, AddSIMD>(x, y, meta, out);
    return out;
}

template <typename T>
inline Tensor<T> sub(const Tensor<T> &x, const Tensor<T> &y) {
   BinaryEwiseMeta meta = make_binary_meta(x, y);
   Tensor<T> out = init_out_from_meta(x, y, meta);
   ewise::binary_ewise_tag<T, SubtractSIMD>(x, y, meta, out);
   return out;
}

template <typename T>
inline Tensor<T> mul(const Tensor<T> &x, const Tensor<T> &y) {
   BinaryEwiseMeta meta = make_binary_meta(x, y);
   Tensor<T> out = init_out_from_meta(x, y, meta);
   ewise::binary_ewise_tag<T, MultiplySIMD>(x, y, meta, out);
   return out;
}

template <typename T>
inline Tensor<T> div(const Tensor<T> &x, const Tensor<T> &y) {
   BinaryEwiseMeta meta = make_binary_meta(x, y);
   Tensor<T> out = init_out_from_meta(x, y, meta);
   ewise::binary_ewise_tag<T, DivideSIMD>(x, y, meta, out);
   return out;
}

template <typename T>
inline Tensor<T> pow(const Tensor<T> &x, const Tensor<T> &y) {
   BinaryEwiseMeta meta = make_binary_meta(x, y);
   Tensor<T> out = init_out_from_meta(x, y, meta);
   ewise::binary_ewise_tag<T, PowerSIMD>(x, y, meta, out);
   return out;
}

} // namespace math

#endif // EWISE_H
