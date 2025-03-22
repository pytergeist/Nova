#pragma once
#include "tensor.h"

// Constructor definition.
template <typename T>
Tensor<T>::Tensor(const std::vector<T>& data) : arr(data) {}

template <typename T>
Tensor<T> Tensor<T>::operator+(const Tensor<T> &tensor) const {
    if (arr.size() != tensor.arr.size()) {
        throw std::invalid_argument("Tensor sizes do not match");
    }
    std::vector<T> new_arr(tensor.arr.size());
    for (size_t i = 0; i < this->arr.size(); ++i) {
        new_arr[i] = this->arr[i] + tensor.arr[i];
    }
    return Tensor<T>(new_arr);
}

template <typename T>
Tensor<T> Tensor<T>::operator*(const Tensor<T> &tensor) const {
    if (arr.size() != tensor.arr.size()) {
        throw std::invalid_argument("Tensor sizes do not match");
    }
    std::vector<T> new_arr(tensor.arr.size());
    for (size_t i = 0; i < this->arr.size(); ++i) {
        new_arr[i] = this->arr[i] * tensor.arr[i];
    }
    return Tensor<T>(new_arr);
}

template <typename T>
Tensor<T> Tensor<T>::operator-(const Tensor<T> &tensor) const {
    if (arr.size() != tensor.arr.size()) {
        throw std::invalid_argument("Tensor sizes do not match");
    }
    std::vector<T> new_arr(tensor.arr.size());
    for (size_t i = 0; i < this->arr.size(); ++i) {
        new_arr[i] = this->arr[i] - tensor.arr[i];
    }
    return Tensor<T>(new_arr);
}

template <typename T>
Tensor<T> Tensor<T>::operator/(const Tensor<T> &tensor) const {
    if (arr.size() != tensor.arr.size()) {
        throw std::invalid_argument("Tensor sizes do not match");
    }
    std::vector<T> new_arr(tensor.arr.size());
    for (size_t i = 0; i < this->arr.size(); ++i) {
        new_arr[i] = this->arr[i] / tensor.arr[i];
    }
    return Tensor<T>(new_arr);
}
