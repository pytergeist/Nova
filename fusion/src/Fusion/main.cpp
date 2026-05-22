#include "Fusion/simulation/Potentials/NonBonded.hpp"
#include "Fusion/simulation/core/Neighbours.hpp"
#include "Fusion/simulation/core/ParticleState.hpp"
#include "Fusion/simulation/core/TopoIter.hpp"
#include "Fusion/simulation/cpu/pairwise/PairwiseTraits.hpp"
#include "core/tensor/RawTensor.hpp"

#include "Fusion/simulation/autodiff/ADSimulation.hpp"
#include "Fusion/simulation/core/InteractionIR.h"
#include "Fusion/simulation/core/InteractionPlan.h"
#include "Fusion/simulation/core/InteractionPlanMeta.hpp"
#include "Fusion/simulation/ops/GatherIndex.hpp"

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

std::string print_format(PairIndexFormat format) {
   std::ostringstream oss;
   if (format == PairIndexFormat::PairBlockedCRS) {
      oss << "PairBlockedCRS" << std::endl;
   }
   return oss.str();
};

std::string print_layout(ParticleLayout layout) {
   std::ostringstream oss;
   if (layout == ParticleLayout::AoSoA) {
      oss << "AoSoA" << std::endl;
   }
   return oss.str();
}

int main() {
   using T = float;

   constexpr std::size_t N = 8;
   constexpr std::size_t DIM = 3;
   constexpr std::size_t TILE = 4;

   // using Layout = ParticlesAoSoA<T, DIM, TILE>;
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
   AoSoATensor<T> aosoa{X, static_cast<std::size_t>(4)};
   // aosoa.assign_component_major(X);

   using Layout = ParticleField<T>;

   Layout field{aosoa};

   // Layout psoa = Layout::from_three_n_raw_tensor(8, X, X, X, X);

   //   LJParams<T> params{0.2f, 0.7f};
   // NoParams params;

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
   NoParams params;
   GatherIndexMeta<T, Layout> meta =
       construct_gather_index_meta<T, Layout>(field, edges);
   ADTensor<T> x_diff{field.x().raw()};
   ADTensor<T> out = pair_delta3<T, Layout>(x_diff, field, meta, params);
   std::cout << out << std::endl;

   return 0;
};
