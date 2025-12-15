#define ANKERL_NANOBENCH_IMPLEMENT

#include <nanobench.h>
#include <random>
#include <vector>

#include "Fusion/core/TensorBase.hpp"

std::vector<float> make_random_float_vector(std::size_t N, unsigned seed,
                                            float min = 0, float max = 100) {
   std::mt19937 engine{seed};
   std::uniform_real_distribution<float> dist{min, max};

   std::vector<float> v(N);
   std::generate(v.begin(), v.end(), [&]() { return dist(engine); });
   return v;
}

int main() {
   unsigned seed = 123456789;
   int epoch_iterations = 10000;
   int milisecs = 200;

   std::vector<std::size_t> sizes = {2, 4, 8, 16, 32, 64, 128, 256};

   ankerl::nanobench::Bench bench;
   bench.title("Fusion");

   for (auto size : sizes) {
      auto v1 = make_random_float_vector(size * size, seed);

      std::vector<std::size_t> shape = {size, size};

      TensorBase<float> t1(shape, v1, DType::FLOAT32,
                           Device{DeviceType::CPU, 0});
      TensorBase<float> t2(shape, v1, DType::FLOAT32,
                           Device{DeviceType::CPU, 0});

      bench.minEpochIterations(epoch_iterations)
          .minEpochTime(std::chrono::milliseconds(milisecs))
          .run("Add", [&] { TensorBase<float> t3 = t1 + t2; });

      bench.minEpochIterations(epoch_iterations)
          .minEpochTime(std::chrono::milliseconds(milisecs))
          .run("AddNoOpti", [&] {
             TensorBase<float> t3 = t1 + t2;
             ankerl::nanobench::doNotOptimizeAway(t3);
          });

      bench.minEpochIterations(epoch_iterations)
          .minEpochTime(std::chrono::milliseconds(milisecs))
          .run("Sub", [&] { TensorBase<float> t3 = t1 - t2; });

      bench.minEpochIterations(epoch_iterations)
          .minEpochTime(std::chrono::milliseconds(200))
          .run("SubNoOpti", [&] {
             TensorBase<float> t3 = t1 - t2;
             ankerl::nanobench::doNotOptimizeAway(t3);
          });

      bench.minEpochIterations(epoch_iterations)
          .minEpochTime(std::chrono::milliseconds(milisecs))
          .run("Div", [&] { TensorBase<float> t3 = t1 / t2; });

      bench.minEpochIterations(epoch_iterations)
          .minEpochTime(std::chrono::milliseconds(milisecs))
          .run("DivNoOpti", [&] {
             TensorBase<float> t3 = t1 / t2;
             ankerl::nanobench::doNotOptimizeAway(t3);
          });

      bench.minEpochIterations(epoch_iterations)
          .minEpochTime(std::chrono::milliseconds(milisecs))
          .run("Mul", [&] { TensorBase<float> t3 = t1 * t2; });

      bench.minEpochIterations(epoch_iterations)
          .minEpochTime(std::chrono::milliseconds(milisecs))
          .run("MulNoOpti", [&] {
             TensorBase<float> t3 = t1 * t2;
             ankerl::nanobench::doNotOptimizeAway(t3);
          });

      bench.minEpochIterations(epoch_iterations)
          .minEpochTime(std::chrono::milliseconds(milisecs))
          .run("MatMul", [&] { TensorBase<float> t3 = t1.matmul(t2); });

      bench.minEpochIterations(epoch_iterations)
          .minEpochTime(std::chrono::milliseconds(milisecs))
          .run("MatMulNoOpti", [&] {
             TensorBase<float> t3 = t1.matmul(t2);
             ankerl::nanobench::doNotOptimizeAway(t3);
          });
   }

   return 0;
}
