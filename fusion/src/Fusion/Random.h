#ifndef TENSOR_RANDOM_H
#define TENSOR_RANDOM_H

#include "./Tensor.h"
#include <algorithm>
#include <numeric>
#include <random>
#include <vector>

inline std::mt19937 engine{std::random_device{}()};
// std::mt19937 &engine

template <typename T>
Tensor<T> uniform(const std::vector<size_t> &shape, T min, T max) {
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
#endif
