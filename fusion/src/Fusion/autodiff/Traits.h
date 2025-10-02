#ifndef TRAITS_H
#define TRAITS_H

#include <cstdint>
#include <vector>
#include <initializer_list>

// TODO: Create fixed size multitensor for hot paths


struct ValueID { uint16_t idx; };
struct NodeID  { int16_t  idx; };

template <typename T>
struct MultiTensor {
  std::vector<std::vector<T>> data;

  MultiTensor() = default;

  explicit MultiTensor(std::size_t n) : data(n) {}

  MultiTensor(std::initializer_list<std::initializer_list<T>> init) {
    data.reserve(init.size());
    for (const auto& il : init) {
      data.emplace_back(il.begin(), il.end());
    }
  }

  MultiTensor(std::initializer_list<std::vector<T>> init) : data(init) {}

  void push_back(std::vector<T> v) { data.push_back(std::move(v)); }

  std::size_t size() const noexcept { return data.size(); }

  std::vector<T>&       operator[](std::size_t i)       { return data.at(i); }
  const std::vector<T>& operator[](std::size_t i) const { return data.at(i); }

  auto begin()       { return data.begin(); }
  auto end()         { return data.end();   }
  auto begin() const { return data.begin(); }
  auto end()   const { return data.end();   }
};

#endif // TRAITS_H
