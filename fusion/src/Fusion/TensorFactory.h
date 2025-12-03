#ifndef TENSOR_FACTORY_H
#define TENSOR_FACTORY_H

#include <numeric>
#include <vector>

#include "Fusion/common/Checks.h"
#include "Fusion/storage/StorageInterface.h"
#include "Fusion/core/DTypes.h"

#include "Fusion/Tensor.h"
// Still in Fusion namespace

template <typename T>
Tensor<T> fill(const std::vector<size_t> &shape, T value) {
   size_t n = std::accumulate(shape.begin(), shape.end(), size_t{1},
                              std::multiplies<size_t>());
   std::vector<T> data(n, value);
   return  ADTensor<T>(
    Tensor<T>(shape, std::move(data), DType::Float32, Device::CPU), false
    );
}

template<typename T>
ADTensor<T> zeros(const std::vector<size_t>& shape) { return fill<T>(shape, T(0)); }

template<typename T>
ADTensor<T> ones(const std::vector<size_t>& shape) { return fill<T>(shape, T(1)); }

template<typename T>
ADTensor<T> zeros_like(const ADTensor<T>& other) { return zeros<T>(other.shape()); }

template<typename T>
ADTensor<T> ones_like(const ADTensor<T>& other) { return ones<T>(other.shape()); }


#endif
