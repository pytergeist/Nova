#ifndef TENSOR_FACTORY_H
#define TENSOR_FACTORY_H

#include <numeric>
#include <vector>

#include "Fusion/Tensor.h"
#include "Fusion/common/Checks.hpp"
#include "Fusion/core/DType.h"
#include "Fusion/storage/StorageInterface.hpp"

template <typename T>
RawTensor<T> fill(const std::vector<size_t> &shape, T value, Device device) {
   size_t n = std::accumulate(shape.begin(), shape.end(), size_t{1},
                              std::multiplies<size_t>());
   std::vector<T> data(n, value);
   return RawTensor<T>(shape, std::move(data), DType::FLOAT32, device);
}

template <typename T>
RawTensor<T> zeros(const std::vector<size_t> &shape, Device device) {
   return fill<T>(shape, T(0), device);
}

template <typename T>
RawTensor<T> ones(const std::vector<size_t> &shape, Device device) {
   return fill<T>(shape, T(1), device);
}

template <typename T> RawTensor<T> zeros_like(const RawTensor<T> &other) {
   return zeros<T>(other.shape(), other.device());
}

template <typename T> RawTensor<T> ones_like(const RawTensor<T> &other) {
   return ones<T>(other.shape(), other.device());
}

template <typename T> ADTensor<T> ad_zeros_like(const ADTensor<T> &other) {
   RawTensor<T> raw = zeros_like(other.raw());
   return ADTensor<T>(std::move(raw), other.requires_grad());
}

template <typename T>
ADTensor<T> ad_zeros(const std::vector<size_t> &shape, Device device,
                     bool requires_grad = false) {
   RawTensor<T> raw = zeros<T>(shape, device);
   return ADTensor<T>(std::move(raw), requires_grad);
}

template <typename T> ADTensor<T> ad_ones_like(const ADTensor<T> &other) {
   RawTensor<T> raw = ones_like(other.raw());
   return ADTensor<T>(std::move(raw), other.requires_grad());
}

template <typename T>
ADTensor<T> ad_ones(const std::vector<size_t> &shape, Device device,
                    bool requires_grad = false) {
   RawTensor<T> raw = ones<T>(shape, device);
   return ADTensor<T>(std::move(raw), requires_grad);
}

#endif
