#ifndef EIGEN_HELPERS_H
#define EIGEN_HELPERS_H

#include "../tensor/tensor.h"
#include <functional>

template <typename T> bool is_scalar(const Tensor<T> &x) {
  // rank-0: shape_ was set to {} by your scalar constructor
  if (x.shape().empty())
    return true;
  // or fallback: any single-element tensor
  auto *s = dynamic_cast<const EigenTensorStorage<T> *>(x.storage.get());
  return (s->rows() * s->cols() == 1);
}

namespace tensor_detail {
template <typename T, typename BinaryOp>
Tensor<T> binary_elementwise_op(const Tensor<T> &a, const Tensor<T> &b,
                                BinaryOp op, const char *op_name) {
  // TODO: implement proper shape checking for matrix/scalars
  // if (a.storage->rows() != b.storage->rows() ||
  //     a.storage->cols() != b.storage->cols() ||
  //     a.storage->cols() != b.storage->cols() ||
  //     a.storage->device() != b.storage->device()) {
  //   throw std::invalid_argument(
  //       std::string("Tensor shape/device mismatch in ") + op_name);
  // }

  auto *lhs = dynamic_cast<const EigenTensorStorage<T> *>(a.storage.get());
  auto *rhs = dynamic_cast<const EigenTensorStorage<T> *>(b.storage.get());
  if (!lhs || !rhs) {
    throw std::invalid_argument(std::string("Unsupported storage type for ") +
                                op_name);
  }

  if (is_scalar(a) && !is_scalar(b)) { // TODO: abstract this method
    T scalar = lhs->matrix(0, 0);
    auto rows = rhs->matrix.rows();
    auto cols = rhs->matrix.cols();
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
        constMatrix =
            Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic,
                          Eigen::RowMajor>::Constant(rows, cols, scalar);
    auto resultMat = op(constMatrix, rhs->matrix);
    return Tensor<T>(resultMat);
  }

  if (is_scalar(b) && !is_scalar(a)) {
    T scalar = rhs->matrix(0, 0);
    auto rows = lhs->matrix.rows();
    auto cols = lhs->matrix.cols();
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
        constMatrix =
            Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic,
                          Eigen::RowMajor>::Constant(rows, cols, scalar);
    auto resultMat = op(lhs->matrix, constMatrix);
    return Tensor<T>(resultMat);
  }

  auto resultMat = op(lhs->matrix, rhs->matrix);
  return Tensor<T>(resultMat);
}

template <typename T, typename UnaryOp>
Tensor<T> unary_elementwise_op(const Tensor<T> &a, UnaryOp op,
                               const char *op_name) {
  auto *src = dynamic_cast<const EigenTensorStorage<T> *>(a.storage.get());
  if (!src) {
    throw std::runtime_error(std::string("Unsupported storage type for ") +
                             op_name);
  }
  auto resultMat = op(src->matrix);
  return Tensor<T>(resultMat);
}
} // namespace tensor_detail

#endif // EIGEN_HELPERS_H
