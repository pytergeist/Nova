#ifndef TRAITS_H
#define TRAITS_H

#include <cstdint>

struct ValueID {uint16_t idx;};
struct NodeID {int16_t idx;};

template <typename T>
struct MultiTensor {
  std::vector<std::vector<T>> data;

  MultiTensor() = default;
  explicit MultiTensor(std::size_t n) : data(n) {};

  std::size_t size() const noexcept { return data.size(); };
  std::vector<T>& operator[](std::size_t i) { return data.at(i); };
  const std::vector<T>& operator[](std::size_t i) const noexcept { return data.at(i); };


  auto begin() {return data.begin();};
  auto end() {return data.begin();};
  auto begin() const {return data.begin();};
  auto end() const {return data.begin();};

};

#endif // TRAITS_H
