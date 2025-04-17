#pragma once

#include "../tensor/tensor.h"
#include <algorithm>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace py = pybind11;

namespace tensor_py_helpers {

inline Tensor<double>
make_tensor_from_shape_and_list(const std::vector<size_t> &shape,
                                const std::vector<double> &values) {
  if (shape.size() != 2) {
    throw std::runtime_error(
        "Shape must be a vector of 2 elements [rows, cols]");
  }
  size_t rows = shape[0], cols = shape[1];
  if (values.size() != rows * cols) {
    throw std::runtime_error("Number of values does not match provided shape");
  }
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> mat(
      rows, cols);
  std::copy(values.begin(), values.end(), mat.data());
  return Tensor<double>(mat);
}

inline Tensor<double> make_tensor_from_scalar(double v) {
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> mat(1,
                                                                             1);
  mat(0, 0) = v;
  return Tensor<double>(mat);
}

inline py::array_t<double> tensor_to_numpy(const Tensor<double> &t) {
  const auto *cpuStorage =
      dynamic_cast<const EigenTensorStorage<double> *>(t.storage.get());
  if (!cpuStorage) {
    throw std::runtime_error("Tensor does not use EigenTensorStorage");
  }
  size_t rows = cpuStorage->rows();
  size_t cols = cpuStorage->cols();

  std::vector<py::ssize_t> shape;
  std::vector<py::ssize_t> strides;
  if (cols == 1 && rows > 1) {
    shape = {py::ssize_t(rows)};
    strides = {static_cast<py::ssize_t>(sizeof(double))};
  } else if (rows == 1 && cols > 1) {
    shape = {py::ssize_t(cols)};
    strides = {static_cast<py::ssize_t>(sizeof(double))};
  } else {
    shape = {py::ssize_t(rows), py::ssize_t(cols)};
    strides = {py::ssize_t(cols * sizeof(double)), py::ssize_t(sizeof(double))};
  }

  py::array_t<double> arr(shape, strides);
  auto buf = arr.request();
  double *ptr = static_cast<double *>(buf.ptr);
  std::copy(cpuStorage->matrix.data(), cpuStorage->matrix.data() + rows * cols,
            ptr);
  return arr;
}

} // namespace tensor_py_helpers
