#include <pybind11/cast.h>
#include <pybind11/detail/common.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>

#include "Fusion/autodiff/AutodiffBridge.hpp"
#include "Fusion/autodiff/AutodiffMode.hpp"
#include "Fusion/autodiff/EngineContext.hpp"

#include "factory/BindFactory.h"
#include "random/BindRandom.h"
#include "tensor/BindTensor.h"

namespace py = pybind11;

PYBIND11_MODULE(fusion, m_ten) {
   m_ten.doc() =
       "Fusion Tensor module exposing Tensor<float> (for composition)";
   bind_tensor<float>(m_ten, "Tensor");
   bind_factory<float>(m_ten, "factory");
   bind_random<float>(m_ten, "Random");

   py::class_<Device>(m_ten, "CppDevice")
       .def(py::init<DeviceType, DeviceIdx>(), py::arg("type"),
            py::arg("index") = -1);

   py::enum_<DeviceType>(m_ten, "CppDeviceType")
       .value("CPU", DeviceType::CPU)
       .value("CUDA", DeviceType::CUDA)
       .value("METAL", DeviceType::METAL);

   py::enum_<DType>(m_ten, "CppDType")
       .value("FLOAT32", DType::FLOAT32)
       .value("FLOAT64", DType::FLOAT64)
       .value("INT32", DType::INT32)
       .value("INT64", DType::INT64)
       .value("BOOL", DType::BOOL);

   py::class_<EngineScope<float>>(m_ten, "grad_tape")
       .def(py::init<>())
       .def(
           "__enter__",
           [](EngineScope<float> &self) -> EngineScope<float> & {
              self.enter();
              return self;
           },
           py::return_value_policy::reference)
       .def("__exit__",
            [](EngineScope<float> &self, const py::object &, const py::object &,
               const py::object &) -> bool {
               self.exit();
               return false;
            });

   auto m_ad = m_ten.def_submodule("autodiff", "Autodiff control");

   m_ad.def(
       "enabled",
       [](pybind11::object &state) -> bool {
          if (!state.is_none()) {
             const bool active = pybind11::cast<bool>(state);
             set_autodiff_enabled<float>(active);
          }
          return autodiff::grad_enabled();
       },
       pybind11::arg("state") = pybind11::none(),
       "Get or set whether autodiff is enabled for this thread. "
       "When enabled, a default Engine is installed in the EngineContext.");
}
