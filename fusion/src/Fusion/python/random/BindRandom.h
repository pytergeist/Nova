#ifndef BIND_RANDOM_H
#define BIND_RANDOM_H

#include <pybind11/pybind11.h>
#include <string>
#include <typeinfo>

#include "Fusion/Random.h"

namespace py = pybind11;

template <typename T> void bind_random(py::module_ &m_ten, const char *name) {
   using ADT = ADTensor<T>;
   using Base = TensorBase<T>;
   using RNG = Random<T>;

   py::class_<RNG>(m_ten, name)
       .def(py::init<uint32_t>(), py::arg("seed") = std::random_device{}())
       .def(
           "uniform_cpp",
           [](RNG &self, const std::vector<size_t> &shape, T min, T max) {
              Base b = self.uniform_base(shape, min, max);
              return ADT(std::move(b), false);
           },
           py::arg("shape"), py::arg("min"), py::arg("max") );
}

#endif // BIND_RANDOM_H
