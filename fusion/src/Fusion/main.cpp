#include "Fusion/core/RawTensor.hpp"
#include "Fusion/physics/core/PhysicsIter.hpp"
#include "Fusion/physics/core/State.hpp"
#include "Fusion/physics/cpu/pairwise/PairwiseTraits.hpp"
#include "Fusion/physics/core/Neighbours.hpp"
#include "Fusion/physics/Potentials/NonBonded.hpp"

#include "Fusion/physics/autodiff/ADPhysics.hpp"

std::string shape_str(std::vector<size_t> shape) {
   std::ostringstream oss;
   oss << '(';
   for (size_t i = 0; i < shape.size(); ++i) {
      oss << shape[i];
      if (i + 1 < shape.size())
         oss << ',';
   }
   oss << ')';
   return oss.str();
}

int main() {
   using T = float;

   constexpr std::size_t N = 8;
   constexpr std::size_t DIM = 3;
   constexpr std::size_t TILE = 4;

   using Layout = ParticlesAoSoA<T, DIM, TILE>;
   RawTensor<T> X({(std::int64_t)DIM, (std::int64_t)8},
                  {
                      // x
                      0.0f,
                      1.2f,
                      0.0f,
                      1.2f,
                      0.0f,
                      1.2f,
                      0.0f,
                      1.2f,
                      // y
                      0.0f,
                      0.0f,
                      1.2f,
                      1.2f,
                      0.0f,
                      0.0f,
                      1.2f,
                      1.2f,
                      // z
                      0.0f,
                      0.0f,
                      0.0f,
                      0.0f,
                      1.2f,
                      1.2f,
                      1.2f,
                      1.2f,
                  },
                  DType::FLOAT32, Device{DeviceType::CPU, 0});

   Layout psoa =
       Layout::from_three_n_raw_tensor(8, X, X, X, X);

   LJParams<T> params{0.2f, 0.7f};
//   NoParams params;

   EdgeList edges{std::vector<uint32_t>{
                      0, 0, 0,    // i=0
                      1, 1,       // i=1
                      2,          // i=2
                      3, 3,       // i=3
                      4, 4, 4, 4, // i=4
                      5, 5,       // i=5
                      6, 6,       // i=6
                      7, 7, 7     // i=7
                  },
                  std::vector<uint32_t>{
                      1, 4, 5,    // j for i=0
                      0, 4,       // j for i=1
                      6,          // j for i=2
                      4, 7,       // j for i=3
                      0, 1, 5, 7, // j for i=4
                      0, 4,       // j for i=5
                      2, 7,       // j for i=6
                      3, 4, 6     // j for i=7
                  }};
     PairwiseMeta<T, Layout> meta =
       make_pairwise_meta<T, Layout>(psoa, edges, 1);

     ADTensor<T> x_diff{psoa.x};

     ADTensor<T> out = lj_energy<T, Layout>(x_diff, psoa, meta, params);
     std::cout << out << std::endl;
//   RawTensor<T> oute = lj_energy<T,  Layout>(psoa, edges, params);
//   RawTensor<T> outf = lj_force<T,  Layout>(psoa, edges, params);

//   std::cout << oute << std::endl;
//   std::cout << outf << std::endl;
//   std::cout << typeid(out).name() << std::endl;
//   RawTensor<T> inv_r2 = out.reciprocal();
//   RawTensor<T> inv_r6 = inv_r2.pow(3);
//   RawTensor<T> sr2 = inv_r2 * (lj_params.sigma * lj_params.sigma);
//   RawTensor<T> sr6 = sr2.pow(3);
//   RawTensor<T> e_pair = (sr6 + sr6 - 1) * 4 * lj_params.epsilon;
//   std::cout << shape_str(out.shape()) << std::endl;
//   std::cout << inv_r2 << std::endl;
//   std::cout << inv_r6 << std::endl;
//   std::cout << sr2 << std::endl;
//   std::cout << sr6 << std::endl;
//   std::cout << e_pair << std::endl;


   return 0;
};
