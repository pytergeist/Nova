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
  if (a.arr.size() > 1) {
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

/* Operator overloads for binary operations (all now work with Tensor<T>
 * arguments) and take one argument (other tensor), the binary operations are
 * applied elementwise between *this (current tensor and other tensor).
 */

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

/* Operator overloads for unary operations.
 * Takes no arguments (), the binary operations are applied elementwise
 * to *this (current tensor).
 */

// sqrt(tensor)
template <typename T> Tensor<T> Tensor<T>::sqrt() const {
  return elementwise_unary_op(*this,
                              [](T base) -> T { return std::sqrt(base); });
}

// exp(tensor)
template <typename T> Tensor<T> Tensor<T>::exp() const {
  return elementwise_unary_op(*this,
                              [](T base) -> T { return std::exp(base); });
}

// log(tensor)
template <typename T> Tensor<T> Tensor<T>::log() const {
  return elementwise_unary_op(*this,
                              [](T base) -> T { return std::log(base); });
}

template <typename T>
Tensor<T> Tensor<T>::matmul(const Tensor<T> &tensor) const {
  if constexpr (is_std_vector<T>::value) {
    size_t const rows = this->arr.size();
    size_t const cols = tensor.arr[0].size();
    std::vector<std::vector<double>> result;
    result.resize(rows);
    for (size_t i = 0; i < rows; i++) {
      result[i].resize(cols);
      for (size_t j = 0; j < cols; j++) {
        result[i][j] = 0;
      }
    }

    for (size_t i = 0; i < rows; i++) {
      for (size_t j = 0; j < cols; j++) {
        for (size_t k = 0; k < this->arr[i].size(); k++) {
          result[i][j] += this->arr[i][k] * tensor.arr[k][j];
        }
      }
    }
    return Tensor<T>(result);
    // } else if (!is_std_vector<T>::value) {
    //     size_t const rows = this->arr.size();
    //     constexpr int cols = 1;
    //     std::vector<std::vector<double> > result;
    //     result.resize(rows);
    //     for (size_t i = 0; i < rows; i++) {
    //         result[i].resize(cols);
    //         for (size_t j = 0; j < cols; j++) {
    //             result[i][j] = 0;
    //         }
    //     }
    //
    //     for (size_t i = 0; i < rows; i++) {
    //         for (size_t j = 0; j < cols; j++) {
    //             for (size_t k = 0; k < this->arr[i].size(); k++) {
    //                 result[i][j] += this->arr[i][k] * tensor.arr[j];
    //             }
    //         }
    //     }
    return Tensor<T>(result);
  } else {
    throw std::invalid_argument("Tensor sizes do not match");
  }
}
