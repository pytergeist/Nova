#ifndef TENSOR_RANDOM_H
#define TENSOR_RANDOM_H

#include <algorithm>
#include <numeric>
#include <random>
#include <vector>

#include "Tensor.h"
#include "core/DTypes.h"

// inline std::mt19937 engine{std::random_device{}()};

template <typename T> class Random {
   std::mt19937 engine_;

 public:
   explicit Random(const uint32_t seed = std::random_device{}()) {
      std::mt19937 engine_{seed};
   }

   ~Random() = default;

   Tensor<T> uniform(const std::vector<size_t> &shape, T min, T max) {
      size_t total =
          std::accumulate(shape.begin(), shape.end(), static_cast<size_t>(1),
                          std::multiplies<>());

      std::vector<T> data;
      data.reserve(total);

      std::uniform_real_distribution<T> dist(min, max);

      for (size_t i = 0; i < total; ++i) {
         data.push_back(dist(this->engine_));
      }

      return Tensor<T>(shape, std::move(data), DType::Float32, Device::CPU); // TODO: Pass in dtype/req_grad
   }
};

// std::mt19937 &engine

// template<typename T>
// Tensor<T> uniform(const std::vector<size_t> &shape, T min, T max) {
//     size_t total = std::accumulate(shape.begin(), shape.end(), size_t(1),
//                                    std::multiplies<>());
//
//     std::vector<T> data;
//     data.reserve(total);
//
//     std::uniform_real_distribution<T> dist(min, max);
//
//     for (size_t i = 0; i < total; ++i) {
//         data.push_back(dist(engine));
//     }
//
//     return Tensor<T>(shape, std::move(data), Device::CPU);
// }
#endif
