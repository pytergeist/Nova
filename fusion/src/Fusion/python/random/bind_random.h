#ifndef BIND_RANDOM_H
#define BIND_RANDOM_H

#include "../../random.h"
#include "../../tensor.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>
#include <typeinfo>

namespace py = pybind11;

template <typename T> void bind_random(py::module_ &m, const char *name) {
  // Build docstring for the submodule
  std::string doc = std::string("Random/distribution functions for Tensor<") +
                    typeid(T).name() + ">";
  std::string("Random/distribution functions for Tensor<") + typeid(T).name() +
      ">";

  auto submod = m.def_submodule(name, doc.c_str());

  submod.def(
      "uniform", &uniform<T>,
      "Create a uniform distribution of a shape between min and max values",
      py::arg("shape"), py::arg("min"), py::arg("max"));
}

#endif // BIND_RANDOM_H
