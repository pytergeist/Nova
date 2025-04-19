#ifndef TENSOR_H
#define TENSOR_H

#include "eigen_tensor.h"
#include "storage_tensor.h"
#include <Eigen/Dense>
#include <cblas.h>
#include <initializer_list>
#include <iostream>
#include <stdexcept>

template <typename T> class Tensor;

template <typename T>
std::ostream &operator<<(std::ostream &os, const Tensor<T> &tensor);

template <typename T> class Tensor {
public:
  std::unique_ptr<ITensorStorage<T>> storage;
  std::vector<size_t> shape_;

  explicit Tensor(size_t rows, size_t cols, Device device = Device::CPU);

  explicit Tensor(const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic,
                                      Eigen::RowMajor>
                      &matrix); // TODO: add shape attribute to nested eigen
                                // matrix constructor

  [[nodiscard]] std::vector<size_t> shape() const;

  void setValues(std::initializer_list<T> values);

  void setValues(std::initializer_list<std::initializer_list<T>> nestedValues);

  friend std::ostream &operator<< <T>(std::ostream &os,
                                      const Tensor<T> &tensor);

  // overload the + operator
  Tensor<T> operator+(const Tensor<T> &other) const;

  // overload the - operator
  Tensor<T> operator-(const Tensor<T> &other) const;

  // overload the / operator
  Tensor<T> operator/(const Tensor<T> &other) const;

  // overload the + operator
  Tensor<T> operator*(const Tensor<T> &other) const;

  Tensor<T> sqrt() const;

  Tensor<T> exp() const;

  Tensor<T> log() const;

  Tensor<T> pow(T exponent) const;

  Tensor<T> pow(const Tensor<T> &exponent) const;

  Tensor<T> sum() const;

  Tensor<T> maximum(const Tensor<T> &other) const;

  Tensor<T> matmul(const Tensor<T> &other) const;

  Tensor<T> transpose() const;
};

#include "tensor_algorithms.ipp"
#include "tensor_arithmetic.ipp"
#include "tensor_constructors.ipp"
#include "tensor_io.ipp"
#include "tensor_reductions.ipp"
#endif // TENSOR_H
