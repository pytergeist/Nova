#include "factory/bind_factory.h"
#include "random/bind_random.h"
#include "tensor/bind_tensor.h"
#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(fusion, m) {
  m.doc() = "Fusion Tensor module exposing Tensor<float> (for composition)";
  bind_tensor<float>(m, "Tensor");
  bind_factory<float>(m, "factory");
  bind_random<float>(m, "random");
}
