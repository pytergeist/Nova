#ifndef UNARY_H
#define UNARY_H

#include "../Tensor.h"
#include "../core/ElementWise.h"
#include <string_view>
#include <vector>

namespace math {

template <typename T> inline Tensor<T> sqrt(const Tensor<T> &x) {
   std::vector<size_t> out_shape;
   std::vector<T> out_data;
   ewise::unary_ewise_tag<T, SqrtSIMD>(x, out_shape, out_data);
   return Tensor(std::move(out_shape), std::move(out_data), Device::CPU,
                 x.requires_grad());
}

template <typename T> inline Tensor<T> log(const Tensor<T> &x) {
   std::vector<size_t> out_shape;
   std::vector<T> out_data;
   ewise::unary_ewise_tag<T, NaturalLogSIMD>(x, out_shape, out_data);
   return Tensor(std::move(out_shape), std::move(out_data), Device::CPU,
                 x.requires_grad());
}

template <typename T> inline Tensor<T> exp(const Tensor<T> &x) {
   std::vector<size_t> out_shape;
   std::vector<T> out_data;
   ewise::unary_ewise_tag<T, ExponentialSIMD>(x, out_shape, out_data);
   return Tensor(std::move(out_shape), std::move(out_data), Device::CPU,
                 x.requires_grad());
}

} // namespace math

#endif // UNARY_H
