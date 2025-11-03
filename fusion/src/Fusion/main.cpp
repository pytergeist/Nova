#include <iostream>
#include <vector>
#include <cstddef>
#include <cstdint>
#include <random>

#include "Tensor.h"
#include "autodiff/Engine.h"
#include "autodiff/EngineContext.h"
#include "core/Device.h"


constexpr size_t num = 2048;
constexpr size_t seed_num = 203;

static auto rand_matrix_flat(size_t rows, size_t cols,
                                    uint32_t seed = seed_num) -> std::vector<float> {
   const size_t ndim = rows * cols;
   std::vector<float> buf(ndim);
   std::mt19937 rng(seed);
   std::uniform_real_distribution<float> dist(0.0F, 1.0F);

   for (size_t i = 0; i < ndim; ++i) {
      buf[i] = dist(rng);
      }
   return buf;
}


auto main() -> int {
   size_t a_seed = 42;
   size_t b_seed = 43;
   auto a_data = rand_matrix_flat(num, num, /*seed=*/a_seed);
   auto b_data = rand_matrix_flat(num, num, /*seed=*/b_seed);
//
//   Tensor<float> A({N, N}, a_data, Device::CPU, true);
//   Tensor<float> B({N, N}, a_data, Device::CPU, true);
//
//   Tensor<float> C = A + B;




   const Tensor<float> A({num, num}, a_data, Device::CPU, false);
   const Tensor<float> B({num, num}, b_data, Device::CPU, false);
   Engine<float> engine;
   EngineContext<float>::set(&engine);

   // Warm up
   volatile float guard = 0.f;
   for (int i = 0; i < 3; ++i) {
      auto C = A + B;
      guard += C[0]; // prevent dead-code elimination if applicable
   }

   // Profile window
   for (int i = 0; i < 50; ++i) {
      auto C = A + B;
      guard += C[0];
   }
   std::cout << (guard != 0.f) << "\n";
}
