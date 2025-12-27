#define ANKERL_NANOBENCH_IMPLEMENT

#include <nanobench.h>
#include <random>
#include <vector>

#include "Fusion/core/RawTensor.hpp"

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

      RawTensor<float> t1(shape, v1, DType::FLOAT32,
                          Device{DeviceType::CPU, 0});
      RawTensor<float> t2(shape, v1, DType::FLOAT32,
                          Device{DeviceType::CPU, 0});

      bench.minEpochIterations(epoch_iterations)
          .minEpochTime(std::chrono::milliseconds(milisecs))
          .run("Add", [&] { RawTensor<float> t3 = t1 + t2; });

      bench.minEpochIterations(epoch_iterations)
          .minEpochTime(std::chrono::milliseconds(milisecs))
          .run("AddNoOpti", [&] {
             RawTensor<float> t3 = t1 + t2;
             ankerl::nanobench::doNotOptimizeAway(t3);
          });

      bench.minEpochIterations(epoch_iterations)
          .minEpochTime(std::chrono::milliseconds(milisecs))
          .run("Sub", [&] { RawTensor<float> t3 = t1 - t2; });

      bench.minEpochIterations(epoch_iterations)
          .minEpochTime(std::chrono::milliseconds(200))
          .run("SubNoOpti", [&] {
             RawTensor<float> t3 = t1 - t2;
             ankerl::nanobench::doNotOptimizeAway(t3);
          });

      bench.minEpochIterations(epoch_iterations)
          .minEpochTime(std::chrono::milliseconds(milisecs))
          .run("Div", [&] { RawTensor<float> t3 = t1 / t2; });

      bench.minEpochIterations(epoch_iterations)
          .minEpochTime(std::chrono::milliseconds(milisecs))
          .run("DivNoOpti", [&] {
             RawTensor<float> t3 = t1 / t2;
             ankerl::nanobench::doNotOptimizeAway(t3);
          });

      bench.minEpochIterations(epoch_iterations)
          .minEpochTime(std::chrono::milliseconds(milisecs))
          .run("Mul", [&] { RawTensor<float> t3 = t1 * t2; });

      bench.minEpochIterations(epoch_iterations)
          .minEpochTime(std::chrono::milliseconds(milisecs))
          .run("MulNoOpti", [&] {
             RawTensor<float> t3 = t1 * t2;
             ankerl::nanobench::doNotOptimizeAway(t3);
          });

      bench.minEpochIterations(epoch_iterations)
          .minEpochTime(std::chrono::milliseconds(milisecs))
          .run("MatMul", [&] { RawTensor<float> t3 = t1.matmul(t2); });

      bench.minEpochIterations(epoch_iterations)
          .minEpochTime(std::chrono::milliseconds(milisecs))
          .run("MatMulNoOpti", [&] {
             RawTensor<float> t3 = t1.matmul(t2);
             ankerl::nanobench::doNotOptimizeAway(t3);
          });
   }

   return 0;
}
