#ifndef EWISE_H
#define EWISE_H

#include <string_view>
#include <vector>
#include "../Tensor.h"
#include "../core/ElementWise.h"
#include "Helpers.h"
#include "../storage/StorageInterface.h"


namespace math {

template <typename T>
inline Tensor<T> add(const Tensor<T> &x, const Tensor<T> &y) {
    std::vector<size_t> out_shape;
    Tensor<T> out = init_bin_out_tensor(x, y);
    ewise::binary_ewise_tag<T, AddSIMD>(x, y, out_shape, out);
    return out;
}

template <typename T>
inline Tensor<T> sub(const Tensor<T> &x, const Tensor<T> &y) {
   std::vector<size_t> out_shape;
   Tensor<T> out = init_bin_out_tensor(x, y);
   ewise::binary_ewise_tag<T, SubtractSIMD>(x, y, out_shape, out);
   return out;
}

template <typename T>
inline Tensor<T> mul(const Tensor<T> &x, const Tensor<T> &y) {
   std::vector<size_t> out_shape;
   Tensor<T> out = init_bin_out_tensor(x, y);
   ewise::binary_ewise_tag<T, MultiplySIMD>(x, y, out_shape, out);
   return out;
}

template <typename T>
inline Tensor<T> div(const Tensor<T> &x, const Tensor<T> &y) {
   std::vector<size_t> out_shape;
   Tensor<T> out = init_bin_out_tensor(x, y);
   ewise::binary_ewise_tag<T, DivideSIMD>(x, y, out_shape, out);
   return out;
}

template <typename T>
inline Tensor<T> pow(const Tensor<T> &x, const Tensor<T> &y) {
   std::vector<size_t> out_shape;
   Tensor<T> out = init_bin_out_tensor(x, y);
   ewise::binary_ewise_tag<T, PowerSIMD>(x, y, out_shape, out);
   return out;
}

} // namespace math

#endif // EWISE_H
