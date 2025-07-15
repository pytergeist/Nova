#ifndef TENSOR_RANDOM_H
#define TENSOR_RANDOM_H

#include "./tensor.h"
#include <algorithm>
#include <functional>
#include <numeric>
#include <random>
#include <vector>

namespace fusion_random {
template <typename T>
Tensor<T> uniform(const std::vector<size_t> &shape, T min, T max,
                  std::mt19937 &engine) {
  size_t total = std::accumulate(shape.begin(), shape.end(), size_t(1),
                                 std::multiplies<>());

  std::vector<T> data;
  data.reserve(total);

  std::uniform_real_distribution<T> dist(min, max);

  for (size_t i = 0; i < total; ++i) {
    data.push_back(dist(engine));
  }

  return Tensor<T>(shape, std::move(data), Device::CPU);
}
} // namespace fusion_random
#endif
