#ifndef UNARY_H
#define UNARY_H

#include "../Tensor.h"
#include "../core/ElementWise.h"
#include <string_view>
#include <vector>

namespace math {

template <typename T> inline Tensor<T> sqrt(const Tensor<T> &x) {
   std::vector<size_t> out_shape;
   Tensor<T> out = init_un_out_tensor(x);
   ewise::unary_ewise_tag<T, SqrtSIMD>(x, out_shape, out);
   return out;
}

template <typename T> inline Tensor<T> log(const Tensor<T> &x) {
   std::vector<size_t> out_shape;
   Tensor<T> out = init_un_out_tensor(x);
   ewise::unary_ewise_tag<T, NaturalLogSIMD>(x, out_shape, out);
   return out;
}

template <typename T> inline Tensor<T> exp(const Tensor<T> &x) {
   std::vector<size_t> out_shape;
   Tensor<T> out = init_un_out_tensor(x);
   ewise::unary_ewise_tag<T, ExponentialSIMD>(x, out_shape, out);
   return out;
}

} // namespace math

#endif // UNARY_H
