#include "bind_tensor.h"
#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(fusion, m) {
  m.doc() = "Fusion Tensor module exposing Tensor<double> (for composition)";
  bind_tensor<float>(m, "Tensor");
}
