#include "factory/BindFactory.h"
#include "random/BindRandom.h"
#include "tensor/BindTensor.h"
#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(fusion, m) {
  m.doc() = "Fusion Tensor module exposing Tensor<float> (for composition)";
  bind_tensor<float>(m, "Tensor");
  bind_factory<float>(m, "factory");
  bind_random<float>(m, "Random");
}
