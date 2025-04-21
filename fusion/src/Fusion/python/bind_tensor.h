// File: bind_tensor.h
#pragma once

#include "../tensor/tensor.h"
#include "helpers.h"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <sstream>

namespace py = pybind11;

template <typename T> void bind_tensor(py::module_ &m, const char *name) {
  using PyT = Tensor<T>;
  auto cls = py::class_<PyT>(m, name);

  cls
      // Constructors
      .def(py::init(&tensor_py_helpers::make_tensor_from_shape_and_list),
           "Construct a Tensor from shape [rows, cols] and a flat list of "
           "values.")
      .def(py::init(&tensor_py_helpers::make_tensor_from_scalar),
           "Construct a Tensor from a scalar.")

      // repr
      .def("__repr__",
           [](const PyT &t) {
             std::ostringstream oss;
             oss << t;
             return oss.str();
           })

      // to_numpy
      .def("to_numpy", &tensor_py_helpers::tensor_to_numpy,
           "Return the tensor as a numpy array (length-1 dimensions are "
           "squeezed).")

      // operator overloads
      .def("__add__", &PyT::operator+)
      .def("__sub__", &PyT::operator-)
      .def("__mul__", &PyT::operator*)
      .def("__truediv__", &PyT::operator/)

      // matrix ops & elementwise functions
      .def("__matmul__", &PyT::matmul, "Matrix multiplication of two Tensors.")
      .def("pow", py::overload_cast<const double>(&PyT::pow, py::const_),
           "Raise each element to a scalar power.")
      .def("pow", py::overload_cast<const PyT &>(&PyT::pow, py::const_),
           "Raise each element to the elementwise power given by another "
           "Tensor.")
      .def("sqrt", &PyT::sqrt, "Element-wise square root.")
      .def("exp", &PyT::exp, "Element-wise exponential.")
      .def("log", &PyT::log, "Element-wise natural logarithm.")
      .def("sum", &PyT::sum, "Sum of all elements.")
      .def("transpose", &PyT::transpose, "Transpose of the Tensor.")
      .def("maximum", &PyT::maximum,
           "Element-wise maximum with another Tensor.")
      .def("shape", &Tensor<T>::shape, "shape of Tensor.")
      .def_property_readonly(
          "dtype", [](const Tensor<T> &) { return py::dtype::of<T>(); },
          "The NumPy dtype of this tensor’s elements")
      .def("diagonal", &Tensor<T>::diagonal, "Diagonal of Tensor.");
}
