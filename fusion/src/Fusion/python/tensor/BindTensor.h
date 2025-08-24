// BindTensor.h
#pragma once

#include "../../Random.h"
#include "../../Tensor.h"
#include "../../TensorFactory.h"
#include "Helpers.h"

#include <numeric>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <sstream>
#include <stdexcept>

namespace py = pybind11;

template <typename T> void bind_tensor(py::module_ &m, const char *name) {
  using PyT = Tensor<T>;

  py::class_<PyT>(m, name)
      // --- constructor(shape: List[int]) → zero‐initialized tensor ---
      .def(py::init([](const std::vector<size_t> &shape) {
             size_t total =
                 std::accumulate(shape.begin(), shape.end(), static_cast<size_t>(1),
                                 std::multiplies<size_t>());
             return new PyT(shape, std::vector<T>(total));
           }),
           py::arg("shape"),
           "Construct a Tensor of given shape, zero‐initialized.")

      // --- constructor(shape, flat_data) ---
      .def(py::init([](const std::vector<size_t> &shape,
                       const std::vector<T> &data) {
             size_t total =
                 std::accumulate(shape.begin(), shape.end(), static_cast<size_t>(1),
                                 std::multiplies<size_t>());
             if (data.size() != total) {
               throw std::invalid_argument("shape* must equal data.size()");
             }
             return new PyT(shape, data);
           }),
           py::arg("shape"), py::arg("data"),
           "Construct a Tensor from a shape list and a flat data list.")

      // --- fill from Python list ---
      .def(
          "set_values",
          [](PyT &t, const std::vector<T> &vals) {
            size_t expected = t.flat_size();
            if (vals.size() != expected) {
              throw std::invalid_argument(
                  "set_values: expected " + std::to_string(expected) +
                  " elements, got " + std::to_string(vals.size()));
            }
            auto &out = t.raw_data();
            std::copy(vals.begin(), vals.end(), out.begin());
          },
          py::arg("values"),
          "Fill the Tensor with a flat list of length prod(shape).")

      // --- shape & size accessors ---
      .def_property_readonly(
          "shape", [](const PyT &t) { return t.shape_; },
          "Returns the shape as a list of ints.")
      .def_property_readonly(
          "size", &PyT::flat_size,
          "Returns total number of elements (product of shape).")

      // --- convert to NumPy array ---
      .def("to_numpy", &tensor_py_helpers::tensor_to_numpy,
           "Return a NumPy array view of the Tensor’s contents.")

      // --- repr for debugging ---
      .def("__repr__",
           [](const PyT &t) {
             std::ostringstream oss;
             oss << t;
             return oss.str();
           })

      // --- elementwise binary ops ---
      .def(
          "__add__",
          [](PyT const &a, PyT const &b) {
            // a + b returns an FFunc<...>, so immediately turn it into
            // a real Tensor via your Tensor(FFunc) ctor:
            return PyT(a + b);
          },
          py::is_operator())
      .def("__sub__", &PyT::operator-)
      .def("__isub__", &PyT::operator-=)
      .def("__rsub__", &PyT::operator-)
      .def("__mul__", &PyT::operator*)
      .def("__truediv__", &PyT::operator/)
      .def("__ge__", &PyT::operator>=)

      // --- matrix multiply ( @ ) ---
      .def("__matmul__", &PyT::matmul, "Matrix multiplication (A @ B)")

      // --- elementwise tensor‐power ---
      .def("__pow__", &PyT::pow, py::arg("other"),
           "Elementwise power: each element raised to corresponding element of "
           "other tensor.")

      // --- unary & other ops ---
      .def("sqrt", &PyT::sqrt)
      .def("exp", &PyT::exp)
      .def("log", &PyT::log)
      .def("sum", &PyT::sum)
      .def("maximum", &PyT::maximum, py::arg("other"))
      .def("transpose", &PyT::transpose,
           "Return a new Tensor that is the transpose of this one.")

      // -- factory methods --
      // NB: Currently these factory functions are bound to the Tensor class not
      // free fn's
      // TODO: Create free function bindings for factory methods
      .def(
          "zeros_like", [](const PyT &self) { return zeros_like<T>(self); },
          "Return a Tensor of zeros with the same shape as this one.")
      .def(
          "ones_like", [](const PyT &self) { return ones_like<T>(self); },
          "Return a Tensor of zeros with the same shape as this one.");
}
