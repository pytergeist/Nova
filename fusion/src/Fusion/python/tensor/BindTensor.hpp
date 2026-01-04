#ifndef BIND_TENSOR_HPP
#define BIND_TENSOR_HPP

#include <numeric>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <sstream>
#include <stdexcept>

#include "Fusion/Random.hpp"
#include "Fusion/Tensor.h"
#include "Fusion/TensorFactory.hpp"
#include "Fusion/core/DType.h"

#include "Helpers.hpp"

namespace py = pybind11;

template <typename T> void bind_tensor(py::module_ &m, const char *name) {
   using PyT = ADTensor<T>; // TODO: python currently only knows about ADTensor
                            // - think about this

   py::class_<PyT>(m, name)
       // --- constructor(shape[, requires_grad=False]) → zero-initialized
       // tensor ---
       .def(py::init([](const std::vector<size_t> &shape, const DType dtype,
                        const Device device, bool requires_grad) {
               size_t total = std::accumulate(shape.begin(), shape.end(),
                                              static_cast<size_t>(1),
                                              std::multiplies<size_t>());
               return new PyT(shape, std::vector<T>(total), dtype, device,
                              requires_grad, /*allocator_*/ nullptr);
            }),
            py::arg("shape"), py::arg("dtype"), py::arg("device"),
            py::arg("requires_grad"),
            "Construct a Tensor of given shape, zero-initialized. "
            "Optionally set requires_grad.")

       // --- constructor(shape, flat_data[, requires_grad=False]) ---
       .def(py::init([](const std::vector<size_t> &shape,
                        const std::vector<T> &data, const DType dtype,
                        const Device device, bool requires_grad) {
               size_t total = std::accumulate(shape.begin(), shape.end(),
                                              static_cast<size_t>(1),
                                              std::multiplies<size_t>());
               if (data.size() != total) {
                  throw std::invalid_argument("shape* must equal data.size()");
               }
               return new PyT(shape, data, dtype, device, requires_grad);
            }),
            py::arg("shape"), py::arg("data"), py::arg("dtype"),
            py::arg("device"), py::arg("requires_grad"),
            "Construct a Tensor from a shape list and a flat data list. "
            "Optionally set requires_grad.")

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
              std::copy(vals.begin(), vals.end(), t.begin());
           },
           py::arg("values"),
           "Fill the Tensor with a flat list of length prod(shape).")

       // --- shape, size & name accessors ---
       .def_property_readonly(
           "shape", [](const PyT &t) { return t.shape(); },
           "Returns the shape as a list of ints.")
       .def_property_readonly(
           "ndim", [](const PyT &t) { return t.rank(); },
           "Number of dimensions.")
       .def_property_readonly(
           "dtype", [](const PyT &) { return py::dtype::of<T>(); },
           "NumPy dtype of the tensor.")
       .def_property_readonly("size", &PyT::flat_size,
                              "Total number of elements (product of shape).")

       .def_property_readonly(
           "name", [](const PyT &t) { return t.name; },
           "Returns the name of the Tensor")

       // --- requires_grad property (read/write) ---
       .def_property("requires_grad", &PyT::requires_grad,
                     &PyT::set_requires_grad, "Requires grad flag")

       // --- convert to NumPy array ---
       .def("to_numpy", &tensor_py_helpers::tensor_to_numpy<T>,
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
           "__add__", [](const PyT &a, const PyT &b) { return a + b; },
           py::is_operator())
       .def(
           "__add__", [](const PyT &a, T b) { return a + b; },
           py::is_operator())

       .def(
           "__sub__", [](const PyT &a, const PyT &b) { return a - b; },
           py::is_operator())
       .def(
           "__sub__", [](const PyT &a, T b) { return a - b; },
           py::is_operator())

       .def(
           "__mul__", [](const PyT &a, const PyT &b) { return a * b; },
           py::is_operator())
       .def(
           "__mul__", [](const PyT &a, T b) { return a * b; },
           py::is_operator())

       .def(
           "__truediv__", [](const PyT &a, const PyT &b) { return a / b; },
           py::is_operator())
       .def(
           "__truediv__", [](const PyT &a, T b) { return a / b; },
           py::is_operator())

       .def(
           "__pow__", [](const PyT &a, const PyT &b) { return a.pow(b); },
           py::is_operator())
       .def(
           "__pow__", [](const PyT &a, T b) { return a.pow(b); },
           py::is_operator())

       .def(
           "__ge__", [](const PyT &a, const PyT &b) { return a >= b; },
           py::is_operator())
       .def(
           "__ge__", [](const PyT &a, T b) { return a >= b; },
           py::is_operator())
       .def(
           "__gt__", [](const PyT &a, const PyT &b) { return a > b; },
           py::is_operator())

       .def(
           "__neg__",
           [](const PyT &t) {
              auto z = zeros_like(t.raw());
              return z - t.raw();
           },
           py::is_operator())

       // -- inplace ops --
       .def(
           "__isub__",
           [](PyT &a, const PyT &b) -> PyT & {
              a.raw() -= b.raw();
              return a;
           },
           py::is_operator())

       // --- Unary / other ops ---
       .def("sqrt", &PyT::sqrt)
       .def("exp", &PyT::exp)
       .def("log", &PyT::log)
       .def("sum", [](const PyT &t) { return t.sum(-1, false); })
       .def("sum", &PyT::sum, py::arg("axis"), py::arg("keepdim") = false)
       .def("mean", [](const PyT &t) { return t.mean(-1, false); })
       .def("mean", &PyT::mean, py::arg("axis"), py::arg("keepdim") = false)

       // --- matrix multiply ( @ ) ---
       .def("__matmul__", &PyT::matmul, "Matrix multiplication (A @ B)")

       // --- power & maximum ---
       .def(
           "__pow__", [](const PyT &a, const PyT &b) { return a.pow(b); },
           py::is_operator())
       .def(
           "__pow__", [](const PyT &a, T b) { return a.pow(b); },
           py::is_operator())
       .def(
           "maximum", [](const PyT &a, const PyT &b) { return a.maximum(b); },
           py::is_operator())
       .def(
           "maximum", [](const PyT &a, T b) { return a.maximum(b); },
           py::is_operator())
       .def("swapaxes", &PyT::swapaxes, py::arg("axis1"), py::arg("axis2"))
       .def("backward", &PyT::backward)
       .def("backward", &PyT::backward)
       .def("get_grad", &PyT::grad)
       .def("rank", &PyT::rank);
}

#endif // BIND_TENSOR_HPP
