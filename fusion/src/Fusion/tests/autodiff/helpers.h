#include "Fusion/autodiff/ADTensor.hpp"
#include "Fusion/autodiff/AutodiffMode.hpp"
#include "Fusion/autodiff/Engine.hpp"
#include "Fusion/autodiff/EngineContext.hpp"

inline ADTensor<float> make_test_tensor(bool requires_grad) {
   return ADTensor<float>({2, 3}, std::vector<float>{1, 2, 3, 4, 5, 6},
                          DType::FLOAT32, Device{DeviceType::CPU, 0},
                          requires_grad);
}

struct EngineContextReset {
   inline ~EngineContextReset() { EngineContext<float>::set(nullptr); }
};