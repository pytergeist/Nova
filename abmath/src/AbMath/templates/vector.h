#ifndef ABMATH_VECTOR_H
#define ABMATH_VECTOR_H

#include <cassert>
#include <cstddef>
#include <iostream>
#include <vector>

namespace abmath {

template <typename T>
class Vector {
public:
  // Create a vector with a given number of elements.
  explicit Vector(std::size_t size) : data_(size) {}

  // New constructor: initialize from a std::vector.
  Vector(const std::vector<T>& values) : data_(values) {}

  // Read/write access for elements.
  T &operator[](std::size_t i) { return data_[i]; }
  const T &operator[](std::size_t i) const { return data_[i]; }

  std::size_t size() const { return data_.size(); }

  void print() const {
    for (const auto &value : data_)
      std::cout << value << " ";
    std::cout << std::endl;
  }

private:
  std::vector<T> data_;
};


// vector vector
template <typename T>
Vector<T> operator+(const Vector<T> &a, const Vector<T> &b) {
  assert(a.size() == b.size() && "Vectors must be the same size");
  Vector<T> result(a.size());
  for (std::size_t i = 0; i < a.size(); ++i)
    result[i] = a[i] + b[i];
  return result;
}

// vector + scalar
template <typename T> Vector<T> operator+(const Vector<T> &a, const T &scalar) {
  Vector<T> result(a.size());
  for (std::size_t i = 0; i < a.size(); ++i)
    result[i] = a[i] + scalar;
  return result;
}

// scalar vector
template <typename T> Vector<T> operator+(const T &scalar, const Vector<T> &a) {
  Vector<T> result(a.size());
  for (std::size_t i = 0; i < a.size(); ++i)
    result[i] = a[i] + scalar;
  return result;
}
} // namespace abmath

#endif
