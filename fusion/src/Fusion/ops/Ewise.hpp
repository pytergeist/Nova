#ifndef EWISE_HPP
#define EWISE_HPP

#include <string_view>
#include <vector>

#include "Fusion/core/ElementWise.hpp"
#include "Fusion/core/EwiseMeta.hpp"
#include "Fusion/core/RawTensor.hpp"

#include "Helpers.hpp"

namespace fusion {

namespace math {

template <typename T>
inline RawTensor<T> add(const RawTensor<T> &x, const RawTensor<T> &y) {
   BinaryEwiseMeta meta = make_binary_meta(x, y);
   RawTensor<T> out = init_out_from_meta(x, y, meta);
   ewise::binary_ewise_tag<T, AddSIMD>(x, y, meta, out);
   return out;
}

template <typename T>
inline RawTensor<T> sub(const RawTensor<T> &x, const RawTensor<T> &y) {
   BinaryEwiseMeta meta = make_binary_meta(x, y);
   RawTensor<T> out = init_out_from_meta(x, y, meta);
   ewise::binary_ewise_tag<T, SubtractSIMD>(x, y, meta, out);
   return out;
}

template <typename T>
inline RawTensor<T> mul(const RawTensor<T> &x, const RawTensor<T> &y) {
   BinaryEwiseMeta meta = make_binary_meta(x, y);
   RawTensor<T> out = init_out_from_meta(x, y, meta);
   ewise::binary_ewise_tag<T, MultiplySIMD>(x, y, meta, out);
   return out;
}

template <typename T>
inline RawTensor<T> div(const RawTensor<T> &x, const RawTensor<T> &y) {
   BinaryEwiseMeta meta = make_binary_meta(x, y);
   RawTensor<T> out = init_out_from_meta(x, y, meta);
   ewise::binary_ewise_tag<T, DivideSIMD>(x, y, meta, out);
   return out;
}

template <typename T>
inline RawTensor<T> pow(const RawTensor<T> &x, const RawTensor<T> &y) {
   BinaryEwiseMeta meta = make_binary_meta(x, y);
   RawTensor<T> out = init_out_from_meta(x, y, meta);
   ewise::binary_ewise_tag<T, PowerSIMD>(x, y, meta, out);
   return out;
}

} // namespace math

} // namespace fusion

#endif // EWISE_H
