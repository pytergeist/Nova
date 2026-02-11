#include "Fusion/core/RawTensor.hpp"
#include "Fusion/physics/core/PhysicsIter.hpp"
#include "Fusion/physics/core/State.hpp"
#include "Fusion/physics/primitives/LJ.hpp"

template <typename T>
inline RawTensor<T> init_out_from_meta(const RawTensor<T> &x,
                                       const PairwiseMeta<T> &m) {
   return RawTensor<T>(m.out_shape, x.dtype(), x.device());
}

int main() {
   using T = float;

   RawTensor<T> X{{3, 4},
                  {// x
                   0.0f, 1.0f, 2.0f, 3.0f,
                   // y
                   0.0f, 0.5f, 1.0f, 1.5f,
                   // z
                   0.0f, 0.0f, 0.0f, 0.0f},
                  DType::FLOAT32,
                  Device{DeviceType::CPU, 0}
   };

   ParticlesSoA<T> psoa{X, X, X, X};

   EdgeList edges{std::vector<uint32_t>{0, 1, 2, 0},
                  std::vector<uint32_t>{1, 2, 3, 2}};


   PairwiseMeta<T> meta = make_pairwise_meta(psoa, edges);

   RawTensor<T> out = init_out_from_meta(psoa.x, meta);

   fusion::physics::iter::pairwise_tag<T, SubtractSIMD>(meta, out);

   std::cout << out << std::endl;

   return 0;
};

