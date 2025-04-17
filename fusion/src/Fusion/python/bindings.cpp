#include "bind_tensor.h"
#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(fusion_math, m) {
  m.doc() = "Fusion Tensor module exposing Tensor<double>";
  bind_tensor<double>(m, "Tensor");
}
