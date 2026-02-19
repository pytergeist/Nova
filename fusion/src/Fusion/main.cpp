#include "Fusion/core/RawTensor.hpp"
#include "Fusion/physics/core/PhysicsIter.hpp"
#include "Fusion/physics/core/State.hpp"
#include "Fusion/physics/cpu/pairwise/PairwiseTraits.hpp"
#include "Fusion/physics/core/Neighbours.hpp"
#include "Fusion/physics/primitives/LJ.hpp"

template <typename T, class ParticlesT>
inline RawTensor<T> init_out_from_meta(const RawTensor<T> &x,
                                       const PairwiseMeta<T, ParticlesT> &m) {
   return RawTensor<T>(m.out_shape, x.dtype(), x.device());
}
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

   ParticlesAoSoA<T, DIM, TILE> psoa =
       ParticlesAoSoA<T, DIM, TILE>::from_three_n_raw_tensor(8, X, X, X, X);

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

   PairwiseMeta<T, ParticlesAoSoA<T, DIM, TILE>> meta =
       make_pairwise_meta<T, ParticlesAoSoA<T, DIM, TILE>>(psoa, edges);

   BlockedCRS bcrs = make_blocked_crs<T, ParticlesAoSoA<T, DIM, TILE>>(psoa, edges);

   RawTensor<T> out = init_out_from_meta(psoa.x, meta);

   fusion::physics::iter::pairwise_tag<T, Vec3GatherSub>(meta, out);
   std::cout << shape_str(out.shape()) << std::endl;
   std::cout << out << std::endl;

   return 0;
};
