#include "Fusion/core/tensor/DenseTensor.hpp"
#include "Fusion/simulation/Potentials/NonBonded.hpp"
#include "Fusion/simulation/core/Neighbours.hpp"
#include "Fusion/simulation/core/ParticleState.hpp"
#include "Fusion/simulation/core/TopoIter.hpp"
#include "Fusion/simulation/cpu/pairwise/PairwiseTraits.hpp"

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
   DenseTensor<T> X({(std::int64_t)DIM, (std::int64_t)8},
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
   std::cout << aosoa.base() << std::endl;

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
   ADTensor<T> x_diff{Tensor<float>::from_aosoa(field.x())};
   ADTensor<T> out = pair_delta3<T, Layout>(x_diff, field, meta, params);
   std::cout << out << std::endl;

   auto &y = out.base().aosoa();
   const auto &raw = y.base();
   const auto *ptr = raw.get_ptr();

   // Adjust this if your member name is different.
   const PairBlockedCRS &crs = meta.plan.topology.crs;

   std::cout << "\n=== pair_delta3 output in BCRS/group order ===\n";

   for (std::size_t ib = 0; ib < field.n_blocks(); ++ib) {
      const std::uint32_t g0 = crs.ib_ptr[ib];
      const std::uint32_t g1 = crs.ib_ptr[ib + 1];

      for (std::uint32_t g = g0; g < g1; ++g) {
         const std::uint32_t jb = crs.jb_idx[g];

         const std::uint32_t k0 = crs.jb_ptr[g];
         const std::uint32_t k1 = crs.jb_ptr[g + 1];

         std::cout << "group g=" << g << " ib=" << ib << " jb=" << jb << " k=["
                   << k0 << "," << k1 << ")\n";

         for (std::uint32_t k = k0; k < k1; ++k) {
            const std::uint32_t il = crs.i_lane[k];
            const std::uint32_t jl = crs.j_lane[k];

            const std::uint32_t i = static_cast<std::uint32_t>(ib * TILE + il);
            const std::uint32_t j = static_cast<std::uint32_t>(jb * TILE + jl);

            const std::size_t block = k / TILE;
            const std::size_t lane = k % TILE;

            const T dx = ptr[block * DIM * TILE + 0 * TILE + lane];
            const T dy = ptr[block * DIM * TILE + 1 * TILE + lane];
            const T dz = ptr[block * DIM * TILE + 2 * TILE + lane];

            std::cout << "  k=" << k << " lane=" << lane << " pair " << i
                      << " -> " << j << " il=" << il << " jl=" << jl
                      << " delta=(" << dx << ", " << dy << ", " << dz << ")\n";
         }
      }
   }

   return 0;
};
