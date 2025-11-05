#ifndef UNARY_H
#define UNARY_H

#include "../Tensor.h"
#include "../core/ElementWise.h"
#include <string_view>
#include <vector>

namespace math {

template <typename T> inline Tensor<T> sqrt(const Tensor<T> &x) {
   UnaryEwiseMeta meta = make_binary_meta(x);
   Tensor<T> out = init_out_from_meta(x, meta);
   ewise::unary_ewise_tag<T, SqrtSIMD>(x, meta, out);
   return out;
}

template <typename T> inline Tensor<T> log(const Tensor<T> &x) {
   UnaryEwiseMeta meta = make_binary_meta(x);
   Tensor<T> out = init_out_from_meta(x, meta);
   ewise::unary_ewise_tag<T, NaturalLogSIMD>(x, meta, out);
   return out;
}

template <typename T> inline Tensor<T> exp(const Tensor<T> &x) {
   UnaryEwiseMeta meta = make_binary_meta(x);
   Tensor<T> out = init_out_from_meta(x, meta);
   ewise::unary_ewise_tag<T, ExponentialSIMD>(x, meta, out);
   return out;
}

} // namespace math

#endif // UNARY_H
