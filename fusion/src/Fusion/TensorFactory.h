#ifndef TENSOR_FACTORY_H
#define TENSOR_FACTORY_H

#include <numeric>
#include <vector>

#include "Fusion/Tensor.h"
#include "Fusion/common/Checks.h"
#include "Fusion/core/DType.h"
#include "Fusion/storage/StorageInterface.h"

template <typename T>
Tensor<T> fill(const std::vector<size_t> &shape, T value, Device device) {
   size_t n = std::accumulate(shape.begin(), shape.end(), size_t{1},
                              std::multiplies<size_t>());
   std::vector<T> data(n, value);
   return ADTensor<T>(Tensor<T>(shape, std::move(data), DType::FLOAT32, device),
                      false);
}

template <typename T>
ADTensor<T> zeros(const std::vector<size_t> &shape, Device device) {
   return fill<T>(shape, T(0), device);
}

template <typename T>
ADTensor<T> ones(const std::vector<size_t> &shape, Device device) {
   return fill<T>(shape, T(1), device);
}

template <typename T> ADTensor<T> zeros_like(const ADTensor<T> &other) {
   return zeros<T>(other.shape(), other.device());
}

template <typename T> ADTensor<T> ones_like(const ADTensor<T> &other) {
   return ones<T>(other.shape(), other.device());
}

#endif
