#ifndef BIND_RANDOM_H
#define BIND_RANDOM_H

#include <pybind11/pybind11.h>
#include <string>
#include <typeinfo>

#include "Fusion/Random.h"

namespace py = pybind11;

template <typename T> void bind_random(py::module_ &m, const char *name) {
   using PyT = Random<T>;
   py::class_<PyT>(m, name)
       .def(py::init([](const uint32_t &seed) { return PyT(seed); }))
       .def(py::init([]() { return PyT(); }))
       .def("uniform_cpp", &PyT::uniform,
            "Create a uniform distribution of a shape between min and max "
            "values",
            py::arg("shape"), py::arg("min"), py::arg("max"));
};

#endif // BIND_RANDOM_H
