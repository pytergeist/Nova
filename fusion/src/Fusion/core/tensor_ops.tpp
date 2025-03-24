#pragma once
#include "tensor.h"
#include <functional> // for std::plus, std::minus, etc.
#include <stdexcept>
#include <vector>

/**
 * @brief Helper function that applies a binary operation element-wise.
 *
 * This function checks for three conditions:
 *   - If both tensors have the same size, it applies op to each pair of
 * corresponding elements.
 *   - If the left tensor is a scalar (size 1), it applies op between its single
 * value and each element of the right tensor.
 *   - If the right tensor is a scalar (size 1), it applies op between each
 * element of the left tensor and that single value. Otherwise, it throws an
 * exception.
 *
 * @tparam T The element type.
 * @tparam BinaryOp The binary operation type.
 * @param a Left tensor.
 * @param b Right tensor.
 * @param op Binary operation to perform.
 * @return Tensor<T> The resulting tensor.
 */

template <typename T, typename UnaryOp>
Tensor<T> elementwise_unary_op(const Tensor<T> &a, UnaryOp op) {
  std::vector<T> result;
  if (a.arr.size() >= 1) {
    result.resize(a.arr.size());
    for (size_t i = 0; i < a.arr.size(); i++) {
      result[i] = op(a.arr[i]);
    }
  } else if (a.arr.size() == 1) {
    result.resize(a.arr.size());
    result[0] = op(a.arr[0]);
  } else {
    throw std::invalid_argument("Tensor sizes do not match");
  }
  return Tensor<T>(result);
}

template <typename T, typename BinaryOp>
Tensor<T> elementwise_binary_op(const Tensor<T> &a, const Tensor<T> &b,
                                BinaryOp op) {
  std::vector<T> result;
  if (a.arr.size() == b.arr.size()) {
    result.resize(a.arr.size());
    for (size_t i = 0; i < a.arr.size(); ++i)
      result[i] = op(a.arr[i], b.arr[i]);
  } else if (a.arr.size() == 1) {
    result.resize(b.arr.size());
    for (size_t i = 0; i < b.arr.size(); ++i)
      result[i] = op(a.arr[0], b.arr[i]);
  } else if (b.arr.size() == 1) {
    result.resize(a.arr.size());
    for (size_t i = 0; i < a.arr.size(); ++i)
      result[i] = op(a.arr[i], b.arr[0]);
  } else {
    throw std::invalid_argument("Tensor sizes do not match");
  }
  return Tensor<T>(result);
}

// Constructor definitions

// Constructor from a vector.
template <typename T>
Tensor<T>::Tensor(const std::vector<T> &data) : arr(data) {}

// Constructor from a scalar.
template <typename T> Tensor<T>::Tensor(const T &value) : arr(1, value) {}

// Operator overloads (all now work with Tensor<T> arguments)

// tensor + tensor
template <typename T>
Tensor<T> Tensor<T>::operator+(const Tensor<T> &tensor) const {
  return elementwise_binary_op(*this, tensor, std::plus<T>());
}

// tensor - tensor
template <typename T>
Tensor<T> Tensor<T>::operator-(const Tensor<T> &tensor) const {
  return elementwise_binary_op(*this, tensor, std::minus<T>());
}

// tensor * tensor
template <typename T>
Tensor<T> Tensor<T>::operator*(const Tensor<T> &tensor) const {
  return elementwise_binary_op(*this, tensor, std::multiplies<T>());
}

// tensor / tensor
template <typename T>
Tensor<T> Tensor<T>::operator/(const Tensor<T> &tensor) const {
  return elementwise_binary_op(*this, tensor, std::divides<T>());
}

// tensor (pow) tensor
template <typename T> Tensor<T> Tensor<T>::pow(const Tensor<T> &tensor) const {
  return elementwise_binary_op(
      *this, tensor, [](T base, T exp) -> T { return std::pow(base, exp); });
}

// sqrt(tensor)
template <typename T> Tensor<T> Tensor<T>::sqrt() const {
  return elementwise_unary_op(*this,
                              [](T base) -> T { return std::sqrt(base); });
}
