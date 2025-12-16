// Helpers.h
#pragma once

#include <pybind11/numpy.h>
#include <stdexcept>
#include <vector>

#include "Fusion/Tensor.h"
#include "Fusion/autodiff/ADTensor.hpp"

namespace py = pybind11;

namespace tensor_py_helpers {

inline py::array_t<float> tensor_to_numpy(const Tensor<float> &t) {
   // Grab the shape vector
   const auto &shape = t.shape();
   size_t ndim = shape.size();
   size_t total = t.flat_size();

   // Build Python-side shape and stride arrays
   std::vector<ssize_t> py_shape(shape.begin(), shape.end());
   std::vector<ssize_t> py_strides(ndim);
   // C‐contiguous: stride of last dim is sizeof(float)
   ssize_t running = sizeof(float);
   for (int i = ndim - 1; i >= 0; --i) {
      py_strides[i] = running;
      running *= static_cast<ssize_t>(shape[i]);
   }

   // Allocate the array
   py::array_t<float> arr(py_shape, py_strides);
   auto buf = arr.request();
   float *dst = static_cast<float *>(buf.ptr);

   // Copy from our flat std::vector<float>
   const auto &src = t.raw_data();
   if (t.size() != total) {
      throw std::runtime_error("tensor_to_numpy: size mismatch");
   }
   std::copy(t.begin(), t.end(), dst);

   return arr;
}

} // namespace tensor_py_helpers
