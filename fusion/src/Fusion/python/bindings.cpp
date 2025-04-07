#include "../core/tensor.h"
#include <pybind11/numpy.h> // for numpy array conversion
#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // for automatic conversion of std::vector
#include <sstream>        // for string stream in __repr__

namespace py = pybind11;

PYBIND11_MODULE(fusion_math, m) {
  m.doc() = "Fusion Tensor module exposing Tensor<double>";

  // Bind Tensor<double>
  py::class_<Tensor<double>>(m, "Tensor")
      .def(py::init<const std::vector<double>, const std::vector<size_t> &>(),
           "Create a Tensor from a list of doubles.")
      .def(py::init<const double &>(),
           "Create a Tensor from a list of doubles.")
      .def("__repr__",
           [](const Tensor<double> &t) {
             std::ostringstream oss;
             oss << t;
             return oss.str();
           })
      // Add a method to convert the tensor data to a numpy array.
      .def(
          "to_numpy",
          [](const Tensor<double> &t) {
            // Convert the tensor shape (std::vector<size_t>) to a vector of
            // py::ssize_t
            std::vector<py::ssize_t> shape(t.shape.begin(), t.shape.end());

            // Compute strides: for row-major order, the stride for dimension i
            // is the product of sizes for dimensions i+1... multiplied by
            // sizeof(double)
            std::vector<py::ssize_t> strides(shape.size());
            py::ssize_t stride = sizeof(double);
            // Iterate in reverse order to compute the stride for each dimension
            for (ssize_t i = shape.size() - 1; i >= 0; --i) {
              strides[i] = stride;
              stride *= shape[i];
            }

            // Create a numpy array with the given shape and strides.
            // This allocates an array of the appropriate size.
            py::array_t<double> np_arr(shape, strides);

            // Copy the tensor's data into the numpy array's buffer.
            auto buf = np_arr.request();
            double *ptr = static_cast<double *>(buf.ptr);
            std::copy(t.arr.begin(), t.arr.end(), ptr);

            return np_arr;
          },
          "Return the tensor as a numpy array with the proper shape.")
      // Overload for Tensor + Tensor
      .def("__add__",
           (Tensor<double>(Tensor<double>::*)(const Tensor<double> &) const) &
               Tensor<double>::operator+,
           "Element-wise addition of two Tensors.")
      // Overload for Tensor + scalar
      .def("__add__",
           (Tensor<double>(Tensor<double>::*)(const double &) const) &
               Tensor<double>::operator+,
           "Element-wise addition of a Tensor and a scalar.")
      // Overload for Tensor - Tensor
      .def("__sub__",
           (Tensor<double>(Tensor<double>::*)(const Tensor<double> &) const) &
               Tensor<double>::operator-,
           "Element-wise subtraction of two Tensors.")
      // Overload for Tensor - scalar
      .def("__sub__",
           (Tensor<double>(Tensor<double>::*)(const double &) const) &
               Tensor<double>::operator-,
           "Element-wise subtraction of a scalar from a Tensor.")
      // Overload for Tensor * Tensor
      .def("__mul__",
           (Tensor<double>(Tensor<double>::*)(const Tensor<double> &) const) &
               Tensor<double>::operator*,
           "Element-wise multiplication of two Tensors.")
      // Overload for Tensor * scalar
      .def("__mul__",
           (Tensor<double>(Tensor<double>::*)(const double &) const) &
               Tensor<double>::operator*,
           "Element-wise multiplication of a Tensor with a scalar.")
      // Overload for Tensor / Tensor
      .def("__truediv__",
           (Tensor<double>(Tensor<double>::*)(const Tensor<double> &) const) &
               Tensor<double>::operator/,
           "Element-wise division of two Tensors.")
      // Overload for Tensor / scalar
      .def("__truediv__",
           (Tensor<double>(Tensor<double>::*)(const double &) const) &
               Tensor<double>::operator/,
           "Element-wise division of a Tensor by a scalar.")

      .def("__matmul__", &Tensor<double>::matmul,
           "Matrix multiplication of two Tensors. "
           "For a 2D Tensor @ 2D Tensor or 2D Tensor @ 1D Tensor.")

      .def("pow", &Tensor<double>::pow,
           "Power of two Tensors. "
           "For a 1D Tensor.pow(1D Tensor) or 1D Tensor.pow(scalar).")

      .def("sqrt", &Tensor<double>::sqrt,
           "square root of a tensor. "
           "For a 1D Tensor.sqrt() or 2D Tensor.sqrt().")

      .def("exp", &Tensor<double>::exp,
           "exponential of a tensor. "
           "For a 1D Tensor.exp() or 2D Tensor.exp().")

      .def("log", &Tensor<double>::log,
           "natural log of a tensor. "
           "For a 1D Tensor.log() or 2D Tensor.log().")

      .def("sum", &Tensor<double>::sum,
           "sum of a tensor. "
           "For a 1D Tensor.sum() or 2D Tensor.sum().")

      .def("transpose", &Tensor<double>::transpose,
           "transpose of a tensor. "
           "For a 2D Tensor.transpose().")

      .def("maximum", &Tensor<double>::maximum,
           "transpose of a tensor. "
           "For a 2D Tensor.maximum().");
}
