#ifndef TENSOR_FACTORY_H
#define TENSOR_FACTORY_H
#include "./tensor.h"
#include <numeric>
#include <vector>
// -- factory methods --

// -- factory functions --

template <typename T>
Tensor<T> fill(const std::vector<size_t> &shape, T value) {
  size_t n = std::accumulate(shape.begin(), shape.end(), size_t{1},
                             std::multiplies<size_t>());
  std::vector<T> data(n, value);
  return Tensor<T>(shape, std::move(data), Device::CPU);
}

template <typename T> Tensor<T> zeros(const std::vector<size_t> &shape) {
  return fill<T>(shape, T(0));
}

template <typename T> Tensor<T> ones(const std::vector<size_t> &shape) {
  return fill<T>(shape, T(1));
}

template <typename T> Tensor<T> zeros_like(const Tensor<T> &other) {
  return zeros<T>(other.shape_);
}

template <typename T> Tensor<T> ones_like(const Tensor<T> &other) {
  return ones<T>(other.shape_);
}

#endif
