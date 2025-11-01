#ifndef OPS_COMPARISON_H
#define OPS_COMPARISON_H

#include <string_view>
#include <vector>
#include "../Tensor.h"
#include "../core/ElementWise.h"
#include "Helpers.h"
#include "../core/Device.h"


namespace math {
template <typename T>
inline Tensor<T> greater(const Tensor<T> &x, const Tensor<T> &y) {
   std::vector<size_t> out_shape;
   Tensor<T> out = init_bin_out_tensor(x, y);
   ewise::binary_ewise_tag<T, GreaterThanSIMD>(x, y, out_shape, out);
   return out;
}

template <typename T>
inline Tensor<T> greater_equal(const Tensor<T> &x, const Tensor<T> &y) {
   std::vector<size_t> out_shape;
   Tensor<T> out = init_bin_out_tensor(x, y);
   ewise::binary_ewise_tag<T, GreaterThanEqualSIMD>(x, y, out_shape, out);
   return out;
}

template <typename T>
inline Tensor<T> maximum(const Tensor<T> &x, const Tensor<T> &y) {
   std::vector<size_t> out_shape;
   Tensor<T> out = init_bin_out_tensor(x, y);
   ewise::binary_ewise_tag<T, MaximumSIMD>(x, y, out_shape, out);
   return out;
}
} // namespace math

#endif // OPS_COMPARISON_H
