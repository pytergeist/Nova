#include <pybind11/pybind11.h>
#include "factory/BindFactory.h"
#include "random/BindRandom.h"
#include "tensor/BindTensor.h"
#include "../../autodiff/EngineContext.h"

namespace py = pybind11;

PYBIND11_MODULE(fusion, m) {
  m.doc() = "Fusion Tensor module exposing Tensor<float> (for composition)";
  bind_tensor<float>(m, "Tensor");
  bind_factory<float>(m, "factory");
  bind_random<float>(m, "Random");

  py::class_<EngineScope<float>>(m, "grad_tape")
    .def(py::init<>())
    .def("__enter__",
         [](EngineScope<float>& self) -> EngineScope<float>& {
           self.enter();
           return self;
         },
         py::return_value_policy::reference)
    .def("__exit__",
         [](EngineScope<float>& self, py::object, py::object, py::object) {
           self.exit();
           return false;
         });

  auto m_ad = m.def_submodule("autodiff", "Autodiff control");

  m_ad.def("enabled",
           [](pybind11::object state) {
             if (!state.is_none()) {
               bool on = pybind11::cast<bool>(state);
               set_autodiff_enabled<float>(on);
             }
             return autodiff::grad_enabled();
           },
           pybind11::arg("state") = pybind11::none(),
           "Get or set whether autodiff is enabled for this thread. "
           "When enabled, a default Engine is installed in the EngineContext.");
}
