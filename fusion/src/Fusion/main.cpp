#include <cstddef>
#include <iostream>
#include <vector>

#include "Tensor.h"
#include "autodiff/Engine.h"
#include "autodiff/EngineContext.h"
#include "autodiff/Graph.h"
#include "autodiff/Node.h"
#include "autodiff/NodeInterface.h"
#include "autodiff/Sort.h"
#include "autodiff/policies/Comparison/Comparison.h"
#include "autodiff/policies/Ewise/Ewise.h"
#include "autodiff/policies/LinAlg/LinAlg.h"
#include "autodiff/policies/Operation.h"
#include "autodiff/policies/Reduction/Reduction.h"
#include "autodiff/policies/Reduction/Sum.h"
#include "autodiff/policies/Shape/Shape.h"
#include "autodiff/policies/Transcendental/Transcendental.h"
#include "storage/TensorBuffer.h"
#include "storage/TensorView.h"

#include <cstddef>
#include <random>
#include <vector>

constexpr size_t N = 2048;

std::vector<float> rand_matrix_flat(size_t rows, size_t cols,
                                    uint32_t seed = 123) {
   const size_t n = rows * cols;
   std::vector<float> buf(n);
   std::mt19937 rng(seed);
   std::uniform_real_distribution<float> dist(0.0f, 1.0f);

   for (size_t i = 0; i < n; ++i)
      buf[i] = dist(rng);
   return buf;
}
int main() {
   auto a_data = rand_matrix_flat(N, N, /*seed=*/42);
   auto b_data = rand_matrix_flat(N, N, /*seed=*/43);
//
//   Tensor<float> A({N, N}, a_data, Device::CPU, true);
//   Tensor<float> B({N, N}, a_data, Device::CPU, true);
//
//   Tensor<float> C = A + B;




   Tensor<float> A({N, N}, a_data, Device::CPU, false);
   Tensor<float> B({N, N}, b_data, Device::CPU, false);
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
