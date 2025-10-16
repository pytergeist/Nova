#ifndef EWISE_H
#define EWISE_H

#include <vector>
#include <string_view>
#include <vector>
#include "../Tensor.h"
#include "../core/ElementWise.h"
#include "Helpers.h"



namespace math {

template <typename T>
inline Tensor<T> add(const Tensor<T> &x, const Tensor<T> &y) {
    std::vector<size_t> out_shape;
    std::vector<T> out_data;
    ewise::binary_ewise_tag<T, AddSIMD>(x, y, out_shape, out_data);
    return Tensor<T>(std::move(out_shape), std::move(out_data), Device::CPU, grad_flow(x, y));
}

template <typename T>
inline Tensor<T> sub(const Tensor<T> &x, const Tensor<T> &y) {
    std::vector<size_t> out_shape;
    std::vector<T> out_data;
    ewise::binary_ewise_tag<T, SubtractSIMD>(x, y, out_shape, out_data);
    return Tensor(std::move(out_shape), std::move(out_data), Device::CPU, grad_flow(x, y));
}


template <typename T>
inline Tensor<T> mul(const Tensor<T> &x, const Tensor<T> &y) {
    std::vector<size_t> out_shape;
    std::vector<T> out_data;
    ewise::binary_ewise_tag<T, MultiplySIMD>(x, y, out_shape, out_data);
    return Tensor(std::move(out_shape), std::move(out_data), Device::CPU, grad_flow(x, y));
}

template <typename T>
inline Tensor<T> div(const Tensor<T> &x, const Tensor<T> &y) {
    std::vector<size_t> out_shape;
    std::vector<T> out_data;
    ewise::binary_ewise_tag<T, DivideSIMD>(x, y, out_shape, out_data);
    return Tensor(std::move(out_shape), std::move(out_data), Device::CPU, grad_flow(x, y));
}


template <typename T>
inline Tensor<T> pow(const Tensor<T> &x, const Tensor<T> &y) {
    std::vector<size_t> out_shape;
    std::vector<T> out_data;
    ewise::binary_ewise_tag<T, PowerSIMD>(x, y, out_shape, out_data);
    return Tensor(std::move(out_shape), std::move(out_data), Device::CPU, grad_flow(x, y));
}

}


#endif // EWISE_H
