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

template <typename T>
void bind_tensor(py::module_ &m, const char *name) {
  using PyT = Tensor<T>;

  py::class_<PyT>(m, name)
      // --- constructor(shape[, requires_grad=False]) → zero-initialized tensor ---
      .def(py::init([](const std::vector<size_t> &shape,
                       bool requires_grad) {
             size_t total = std::accumulate(shape.begin(), shape.end(),
                                            static_cast<size_t>(1),
                                            std::multiplies<size_t>());
             return new PyT(shape, std::vector<T>(total),
                            Device::CPU, requires_grad);
           }),
           py::arg("shape"),
           py::arg("requires_grad"),
           "Construct a Tensor of given shape, zero-initialized. "
           "Optionally set requires_grad.")

      // --- constructor(shape, flat_data[, requires_grad=False]) ---
      .def(py::init([](const std::vector<size_t> &shape,
                       const std::vector<T> &data,
                       bool requires_grad) {
             size_t total = std::accumulate(shape.begin(), shape.end(),
                                            static_cast<size_t>(1),
                                            std::multiplies<size_t>());
             if (data.size() != total) {
               throw std::invalid_argument("shape* must equal data.size()");
             }
             return new PyT(shape, data, Device::CPU, requires_grad);
           }),
           py::arg("shape"), py::arg("data"),
           py::arg("requires_grad"),
           "Construct a Tensor from a shape list and a flat data list. "
           "Optionally set requires_grad.")

      // --- fill from Python list ---
      .def("set_values",
           [](PyT &t, const std::vector<T> &vals) {
             size_t expected = t.flat_size();
             if (vals.size() != expected) {
               throw std::invalid_argument(
                   "set_values: expected " + std::to_string(expected) +
                   " elements, got " + std::to_string(vals.size()));
             }
             std::copy(vals.begin(), vals.end(), t.begin());
           },
           py::arg("values"),
           "Fill the Tensor with a flat list of length prod(shape).")

      // --- shape & size accessors ---
      .def_property_readonly("shape",
           [](const PyT &t) { return t.shape_; },
           "Returns the shape as a list of ints.")
      .def_property_readonly("ndim",
           [](const PyT &t) { return t.rank_; },
           "Number of dimensions.")
      .def_property_readonly("dtype",
           [](const PyT &) { return py::dtype::of<T>(); },
           "NumPy dtype of the tensor.")
      .def_property_readonly("size", &PyT::flat_size,
           "Total number of elements (product of shape).")

      // --- requires_grad property (read/write) ---
      .def_property("requires_grad",
           [](const PyT &t) { return t.requires_grad(); },
           [](PyT &t, bool v) { t.requires_grad_ = v; },
           "Whether this tensor participates in autodiff.")

      // --- convert to NumPy array ---
      .def("to_numpy", &tensor_py_helpers::tensor_to_numpy,
           "Return a NumPy array view of the Tensor’s contents.")

      // --- repr for debugging ---
      .def("__repr__", [](const PyT &t) {
             std::ostringstream oss;
             oss << t;
             return oss.str();
           })

      // --- elementwise binary ops ---
      .def("__add__", [](const PyT &a, const PyT &b) { return PyT(a + b); },
           py::is_operator())
      .def("__sub__", [](const PyT &a, const PyT &b) { return PyT(a - b); },
           py::is_operator())
      .def("__isub__", &PyT::operator-=)
      .def("__sub__", [](const PyT &a, const PyT &b) { return PyT(b - a); },
           py::is_operator())
      .def("__mul__", [](const PyT &a, const PyT &b) { return PyT(a * b); },
           py::is_operator())
      .def("__truediv__", [](const PyT &a, const PyT &b) { return PyT(a / b); },
           py::is_operator())
      .def("__ge__", [](const PyT& a, const PyT& b) { return a >= b; }, py::is_operator())
      .def("__gt__", &PyT::operator>)
      .def("__neg__", [](const PyT &t) {
             auto z = zeros_like<T>(t);
             return z - t;
           }, py::is_operator())

      // --- matrix multiply ( @ ) ---
      .def("__matmul__", &PyT::matmul, "Matrix multiplication (A @ B)")

      // --- elementwise tensor-power ---
      .def("__pow__", &PyT::pow, py::arg("other"),
           "Elementwise power.")

      // --- unary & other ops ---
      .def("sqrt", &PyT::sqrt)
      .def("exp", &PyT::exp)
      .def("log", &PyT::log)
      .def("sum", &PyT::sum)
      .def("maximum", &PyT::maximum, py::arg("other"))
      .def("transpose", &PyT::transpose, "Return the transpose.")
      .def("swapaxes", &PyT::swapaxes, py::arg("axis1"), py::arg("axis2"))
      .def("diag", &PyT::diagonal)
      .def("backward", &PyT::backward)
      .def("get_grad", &PyT::grad)

      // -- factory methods bound on the class for now --
      .def("zeros_like", [](const PyT &self) { return zeros_like<T>(self); },
           "Zeros with same shape.")
      .def("ones_like",  [](const PyT &self) { return ones_like<T>(self); },
           "Ones with same shape.");
}
