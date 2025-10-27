#define ANKERL_NANOBENCH_IMPLEMENT

#include <nanobench.h>
#include <random>
#include <vector>

#include <Eigen/Core>
using namespace Eigen;

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
   bench.title("Eigen");

   for (auto size : sizes) {
      auto v1 = make_random_float_vector(size * size, seed);

      MatrixXf a(size, size);
      MatrixXf b(size, size);
      MatrixXf c = MatrixXf::Zero(size, size);

      size_t k = 0;
      for (size_t i = 0; i < size; i++) {
         for (size_t j = 0; j < size; j++) {
            a(j, i) = k;
            b(j, i) = k;
            k++;
         }
      }

      bench

          .minEpochIterations(epoch_iterations)
          .minEpochTime(std::chrono::milliseconds(milisecs))
          .run("Add", [&] { MatrixXf c = a.array() + b.array(); });

      bench.minEpochIterations(epoch_iterations)
          .minEpochTime(std::chrono::milliseconds(milisecs))
          .run("AddNoOpti", [&] {
             MatrixXf c = a.array() + b.array();
             ankerl::nanobench::doNotOptimizeAway(c);
          });

      bench.minEpochIterations(epoch_iterations)
          .minEpochTime(std::chrono::milliseconds(milisecs))
          .run("Sub", [&] { MatrixXf c = a.array() - b.array(); });

      bench.minEpochIterations(epoch_iterations)
          .minEpochTime(std::chrono::milliseconds(200))
          .run("SubNoOpti", [&] {
             MatrixXf c = a.array() - b.array();
             ;
             ankerl::nanobench::doNotOptimizeAway(c);
          });

      bench.minEpochIterations(epoch_iterations)
          .minEpochTime(std::chrono::milliseconds(milisecs))
          .run("Div", [&] { MatrixXf c = a.array() / b.array(); });

      bench.minEpochIterations(epoch_iterations)
          .minEpochTime(std::chrono::milliseconds(milisecs))
          .run("DivNoOpti", [&] {
             MatrixXf c = a.array() / b.array();
             ankerl::nanobench::doNotOptimizeAway(c);
          });

      bench.minEpochIterations(epoch_iterations)
          .minEpochTime(std::chrono::milliseconds(milisecs))
          .run("Mul", [&] { MatrixXf c = a.array() * b.array(); });

      bench.minEpochIterations(epoch_iterations)
          .minEpochTime(std::chrono::milliseconds(milisecs))
          .run("MulNoOpti", [&] {
             MatrixXf c = a.array() * b.array();
             ankerl::nanobench::doNotOptimizeAway(c);
          });

      bench.minEpochIterations(epoch_iterations)
          .minEpochTime(std::chrono::milliseconds(milisecs))
          .run("MatMul", [&] { MatrixXf c = a * b; });

      bench.minEpochIterations(epoch_iterations)
          .minEpochTime(std::chrono::milliseconds(milisecs))
          .run("MatMulNoOpti", [&] {
             MatrixXf c = a * b;
             ankerl::nanobench::doNotOptimizeAway(c);
          });
   }

   return 0;
}
