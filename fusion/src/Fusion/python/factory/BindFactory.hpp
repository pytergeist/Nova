#ifndef BIND_FACTORY_HPP
#define BIND_FACTORY_HPP

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>
#include <typeinfo>

#include "Fusion/TensorFactory.hpp"

namespace py = pybind11;

// Binds free-function factories (fill, zeros, ones, etc.) for Tensor<T>
// into a submodule of the given module m with the specified name.
// Example: bind_factory<float>(m, "random"); creates submodule m.random
// and adds functions fill, zeros, ones, etc. under fusion.random

template <typename T> void bind_factory(py::module_ &m, const char *name) {
   std::string doc =
       std::string("factory functions for Tensor<") + typeid(T).name() + ">";

   auto submod = m.def_submodule(name, doc.c_str());

   submod.def("fill", &fill<T>, "Create a tensor filled with a given value",
              py::arg("shape"), py::arg("value"), py::arg("device"));
   submod.def("zeros", &zeros<T>, "Create a tensor of zeros", py::arg("shape"),
              py::arg("device"));
   submod.def("ones", &ones<T>, "Create a tensor of ones", py::arg("shape"),
              py::arg("device"));
   submod.def(
       "zeros_like", [](const ADTensor<T> &t) { return ad_zeros_like<T>(t); },
       py::arg("other"));

   submod.def(
       "ones_like", [](const ADTensor<T> &t) { return ad_ones_like<T>(t); },
       py::arg("other"));
}

#endif // BIND_FACTORY_HPP
