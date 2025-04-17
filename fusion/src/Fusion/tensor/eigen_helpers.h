#ifndef EIGEN_HELPERS_H
#define EIGEN_HELPERS_H

#include "tensor.h"
#include <functional>

namespace tensor_detail {

template <typename T, typename BinaryOp>
Tensor<T> binary_elementwise_op(const Tensor<T> &a, const Tensor<T> &b,
                                BinaryOp op, const char *op_name) {
  if (a.storage->rows() != b.storage->rows() ||
      a.storage->cols() != b.storage->cols() ||
      a.storage->device() != b.storage->device()) {
    throw std::invalid_argument(
        std::string("Tensor shape/device mismatch in ") + op_name);
  }

  auto *lhs = dynamic_cast<const EigenTensorStorage<T> *>(a.storage.get());
  auto *rhs = dynamic_cast<const EigenTensorStorage<T> *>(b.storage.get());
  if (!lhs || !rhs) {
    throw std::invalid_argument(std::string("Unsupported storage type for ") +
                                op_name);
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
