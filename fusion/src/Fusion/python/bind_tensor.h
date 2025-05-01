#pragma once

#include "../tensor/tensor.h"
#include "helpers.h"

#include <algorithm>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <sstream>
#include <stdexcept>

namespace py = pybind11;

template <typename T> void bind_tensor(py::module_ &m, const char *name) {
  using PyT = Tensor<T>;

  py::class_<PyT>(m, name)
      // --- constructor from (rows, cols) ---
      .def(py::init<size_t, size_t>(), py::arg("rows"), py::arg("cols"),
           "Construct a Tensor with given shape (rows, cols). Contents "
           "uninitialized.")

      // --- set raw values from a flat Python list ---
      .def(
          "set_values",
          [](PyT &t, const std::vector<T> &vals) {
            size_t expected = t.storage->rows() * t.storage->cols();
            if (vals.size() != expected) {
              throw std::invalid_argument(
                  "set_values: expected " + std::to_string(expected) +
                  " elements, got " + std::to_string(vals.size()));
            }
            std::copy(vals.begin(), vals.end(), t.storage->data());
          },
          py::arg("values"),
          "Fill the Tensor from a flat list of length rows*cols.")

      // --- expose as NumPy array ---
      .def("to_numpy", &tensor_py_helpers::tensor_to_numpy,
           "Return a NumPy array view of the Tensor’s contents "
           "(1-D vectors squeezed).")

      // --- shape getter ---
      .def("shape", &PyT::shape,
           "Return [rows, cols] as a Python list of two ints.")

      // --- repr for debugging ---
      .def("__repr__",
           [](const PyT &t) {
             std::ostringstream oss;
             oss << t;
             return oss.str();
           })

      // --- elementwise binary operators ---
      .def("__add__", &PyT::operator+)
      .def("__sub__",
           py::overload_cast<const PyT &>(&PyT::operator-, py::const_))
      .def("__mul__", &PyT::operator*)
      .def("__truediv__", &PyT::operator/)

      // --- matrix multiply ---
      .def("__matmul__", &PyT::matmul,
           "Matrix multiplication (like @ in NumPy)")
      .def("__pow__", py::overload_cast<T>(&PyT::pow, py::const_),
           py::arg("exponent"), "Raise each element to a scalar power.")

      // ** operator: tensor version
      .def("__pow__", py::overload_cast<const PyT &>(&PyT::pow, py::const_),
           py::arg("exponent"), "Elementwise tensor power.")
      // --- unary ops ---
      .def("__neg__", py::overload_cast<>(&PyT::operator-, py::const_))
      .def("sqrt", &PyT::sqrt)
      .def("exp", &PyT::exp)
      .def("log", &PyT::log)

      // --- reductions & others ---
      .def("sum", &PyT::sum)
      .def("maximum", &PyT::maximum, py::arg("other"))
      .def("transpose", &PyT::transpose)
      .def("diagonal", &PyT::diagonal);
}
