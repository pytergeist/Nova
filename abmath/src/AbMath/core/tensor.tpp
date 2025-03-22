#pragma once
#include "tensor.h"
/*
This file contains template functions for operator overiding simple element wise arithmatic operations
in the tensor class. The current implamentations are for +,-,/,* operators for tensor/tensor and tensor/scalar
operations
 */
// Constructor definition.
template <typename T>
Tensor<T>::Tensor(const std::vector<T>& data) : arr(data) {}

// tensor + tensor
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

// tensor + scalar
template <typename T>
Tensor<T> Tensor<T>::operator+(const T &scalar) const {
    std::vector<T> new_arr(this->arr.size());
    for (size_t i = 0; i < this->arr.size(); ++i) {
        new_arr[i] = this->arr[i] + scalar;
    }
    return Tensor<T>(new_arr);
}


// tensor * tensor
template <typename T>
Tensor<T> Tensor<T>::operator*(const Tensor<T> &tensor) const {
    if (arr.size() != tensor.arr.size()) {
        throw std::invalid_argument("Tensor sizes do not match");
    }
    std::vector<T> new_arr(this->arr.size());
    for (size_t i = 0; i < this->arr.size(); ++i) {
        new_arr[i] = this->arr[i] * tensor.arr[i];
    }
    return Tensor<T>(new_arr);
}

// tensor * scalar
template <typename T>
Tensor<T> Tensor<T>::operator*(const T &scalar) const {
    std::vector<T> new_arr(this->arr.size());
    for (size_t i = 0; i < this->arr.size(); ++i) {
        new_arr[i] = this->arr[i] * scalar;
    }
    return Tensor<T>(new_arr);
}

// tensor - tensor
template <typename T>
Tensor<T> Tensor<T>::operator-(const Tensor<T> &tensor) const {
    if (arr.size() != tensor.arr.size()) {
        throw std::invalid_argument("Tensor sizes do not match");
    }
    std::vector<T> new_arr(this->arr.size());
    for (size_t i = 0; i < this->arr.size(); ++i) {
        new_arr[i] = this->arr[i] - tensor.arr[i];
    }
    return Tensor<T>(new_arr);
}

// tensor - scalar
template <typename T>
Tensor<T> Tensor<T>::operator-(const T &scalar) const {
    std::vector<T> new_arr(this->arr.size());
    for (size_t i = 0; i < this->arr.size(); ++i) {
        new_arr[i] = this->arr[i] - scalar;
    }
    return Tensor<T>(new_arr);
}

// tensor / tensor
template <typename T>
Tensor<T> Tensor<T>::operator/(const Tensor<T> &tensor) const {
    if (arr.size() != tensor.arr.size()) {
        throw std::invalid_argument("Tensor sizes do not match");
    }
    std::vector<T> new_arr(this->arr.size());
    for (size_t i = 0; i < this->arr.size(); ++i) {
        new_arr[i] = this->arr[i] / tensor.arr[i];
    }
    return Tensor<T>(new_arr);
}


// tensor / scalar
template <typename T>
Tensor<T> Tensor<T>::operator/(const T &scalar) const {
    std::vector<T> new_arr(this->arr.size());
    for (size_t i = 0; i < this->arr.size(); ++i) {
        new_arr[i] = this->arr[i] / scalar;
    }
    return Tensor<T>(new_arr);
}
