#ifndef TRAITS_H
#define TRAITS_H

#include <cstdint>
#include <initializer_list>
#include <vector>

#include "../Tensor.h"

// TODO: Create fixed size multitensor for hot paths

struct ValueID {
  uint16_t idx;
};
struct NodeID {
  int16_t idx;
};

// Traits.h
template <typename T>
struct MultiTensor {
  std::vector<Tensor<T>> data;

  MultiTensor() = default;

  explicit MultiTensor(std::size_t n) { data.reserve(n); }

  MultiTensor(const MultiTensor&) = delete;
  MultiTensor& operator=(const MultiTensor&) = delete;
  MultiTensor(MultiTensor&&) noexcept = default;
  MultiTensor& operator=(MultiTensor&&) noexcept = default;

  void push_back(Tensor<T>&& v) { data.emplace_back(std::move(v)); }
  void push_back(const Tensor<T>&) = delete;

  Tensor<T>& at(std::size_t i) { return data.at(i); }
  const Tensor<T>& at(std::size_t i) const { return data.at(i); }

  std::size_t size() const noexcept { return data.size(); }

  Tensor<T>& operator[](std::size_t i) { return data.at(i); }
  const Tensor<T>& operator[](std::size_t i) const { return data.at(i); }

  auto begin() { return data.begin(); }
  auto end() { return data.end(); }
  auto begin() const { return data.begin(); }
  auto end() const { return data.end(); }
};


#endif // TRAITS_H
