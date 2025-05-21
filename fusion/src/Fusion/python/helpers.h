// helpers.h
#pragma once

#include "../tensor.h"
#include <numeric>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <stdexcept>
#include <vector>

namespace py = pybind11;

namespace tensor_py_helpers {

inline py::array_t<double> tensor_to_numpy(const Tensor<double> &t) {
  // Grab the shape vector
  const auto &shape = t.shape_;
  size_t ndim = shape.size();
  size_t total = t.flat_size();

  // Build Python-side shape and stride arrays
  std::vector<ssize_t> py_shape(shape.begin(), shape.end());
  std::vector<ssize_t> py_strides(ndim);
  // C‐contiguous: stride of last dim is sizeof(double)
  ssize_t running = sizeof(double);
  for (int i = ndim - 1; i >= 0; --i) {
    py_strides[i] = running;
    running *= static_cast<ssize_t>(shape[i]);
  }

  // Allocate the array
  py::array_t<double> arr(py_shape, py_strides);
  auto buf = arr.request();
  double *dst = static_cast<double *>(buf.ptr);

  // Copy from our flat std::vector<double>
  const auto &src = t.raw_data();
  if (src.size() != total) {
    throw std::runtime_error("tensor_to_numpy: size mismatch");
  }
  std::copy(src.begin(), src.end(), dst);

  return arr;
}

} // namespace tensor_py_helpers
