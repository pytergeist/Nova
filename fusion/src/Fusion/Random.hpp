#ifndef TENSOR_RANDOM_H
#define TENSOR_RANDOM_H

#include <algorithm>
#include <numeric>
#include <random>
#include <vector>

#include "Fusion/core/DType.h"

#include "Fusion/Tensor.h"

template <typename T> class Random {
   std::mt19937 engine_;

 public:
   explicit Random(uint32_t seed = std::random_device{}())
       : engine_(seed) {} // FIX: seed the member, not a local var

   RawTensor<T> uniform_base(const std::vector<size_t> &shape, T min, T max,
                             Device device) {
      size_t total =
          std::accumulate(shape.begin(), shape.end(), static_cast<size_t>(1),
                          std::multiplies<>());

      std::vector<T> data;
      data.reserve(total);

      std::uniform_real_distribution<T> dist(min, max);

      for (size_t i = 0; i < total; ++i) {
         data.push_back(dist(engine_));
      }

      return RawTensor<T>(shape, std::move(data), DType::FLOAT32, device);
   }
};

#endif
