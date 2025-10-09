#ifndef TENSOR_FACTORY_H
#define TENSOR_FACTORY_H
#include <numeric>
#include <vector>
#include "./Tensor.h"
#include "common/Checks.h"

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
  FUSION_CHECK(other.is_initialised(), "zeros_like on uninitialised tensor");
  return zeros<T>(other.storage->shape());
}

template <typename T> Tensor<T> ones_like(const Tensor<T> &other) {
  FUSION_CHECK(other.is_initialised(), "oness_like on uninitialised tensor");
  return ones<T>(other.storage->shape());
}

#endif
