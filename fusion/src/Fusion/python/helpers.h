#pragma once

#include "../tensor/tensor.h"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace py = pybind11;

namespace tensor_py_helpers {

inline py::array_t<double> tensor_to_numpy(const Tensor<double> &t) {
  // Downcast storage
  auto *cpu = dynamic_cast<const EigenTensorStorage<double> *>(t.storage.get());
  if (!cpu) {
    throw std::runtime_error("tensor_to_numpy: unsupported storage");
  }

  size_t rows = cpu->matrix.rows();
  size_t cols = cpu->matrix.cols();

  // Build shape & strides
  std::vector<ssize_t> shape, strides;
  if (rows == 1 && cols == 1) {
    // scalar
    return py::float_(cpu->matrix(0, 0));
  } else if (rows == 1) {
    shape = {static_cast<ssize_t>(cols)};
    strides = {static_cast<ssize_t>(sizeof(double))};
  } else if (cols == 1) {
    shape = {static_cast<ssize_t>(rows)};
    strides = {static_cast<ssize_t>(sizeof(double))};
  } else {
    shape = {static_cast<ssize_t>(rows), static_cast<ssize_t>(cols)};
    strides = {static_cast<ssize_t>(cols * sizeof(double)),
               static_cast<ssize_t>(sizeof(double))};
  }

  // Create array and copy
  py::array_t<double> arr(shape, strides);
  auto buf = arr.request();
  double *dst = static_cast<double *>(buf.ptr);
  std::copy(cpu->matrix.data(), cpu->matrix.data() + rows * cols, dst);
  return arr;
}

} // namespace tensor_py_helpers
