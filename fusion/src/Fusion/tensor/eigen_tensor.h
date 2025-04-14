#ifndef EIGEN_TENSOR_H
#define EIGEN_TENSOR_H

#include "storage_tensor.h"
#include <Eigen/Dense>

template <typename T> class EigenTensorStorage : public ITensorStorage<T> {
public:
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> matrix;

  EigenTensorStorage(size_t rows, size_t cols) : matrix(rows, cols) {};

  T *data() override { return matrix.data(); }
  const T *data() const override { return matrix.data(); }

  [[nodiscard]] size_t rows() const override { return matrix.rows(); }
  [[nodiscard]] size_t cols() const override { return matrix.cols(); }

  [[nodiscard]] Device device() const override { return Device::CPU; }
};

#endif // EIGEN_TENSOR_H
