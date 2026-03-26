#ifndef FUSION_PHYSICS_AUTODIFF_REGISTRY_PAIRWISE_LJ_HPP
#define FUSION_PHYSICS_AUTODIFF_REGISTRY_PAIRWISE_LJ_HPP

#include <string_view>
#include <vector>

#include "Fusion/autodiff/AutodiffMeta.hpp"
#include "Fusion/autodiff/AutodiffMode.hpp"
#include "Fusion/autodiff/registry/Operation.hpp"
#include "Fusion/common/Checks.hpp"
#include "Fusion/core/RawTensor.hpp"

#include "Fusion/simulation/autodiff/SimParamPlan.hpp"
#include "Fusion/simulation/cpu/pairwise/PairwiseParams.hpp"

template <typename T, class ParticlesT> struct LennardJones {
   static constexpr std::string_view name = "LennardJones";
   using In = AutodiffMeta<T>;
   using Out = AutodiffMeta<T>;
   using GradIn = AutodiffMeta<T>;
   using GradOut = AutodiffMeta<T>;

   using ParamPlan = LJParamPlan<T, ParticlesT>;

   Out forward(Context<T> &context, In &input) {
      const autodiff::NoGradGuard _;
      const RawTensor<T> &x = input.at(0);
      const ParamPlan &p = std::any_cast<const ParamPlan &>(input.op_param);
      PairwiseMeta<T, ParticlesT> meta = p.meta;
      context.save("x", x);
      RawTensor<T> e = lj_energy_from_meta(x, *p.particles, meta, p.params);
      Out out;
      out.push_back(e);
      return out;
   };

   GradIn backward(Context<T> &context, GradOut &grad_out) {
      ////      if (grad_out.empty()) {
      ////         return {};
      ////      }
      ////      const RawTensor<T> &x = context.template
      ///load<RawTensor<T>>("x"); /      const RawTensor<T> &g0 =
      ///grad_out.at(0); /      FUSION_CHECK(!g0.empty(), "MatMul::backward:
      ///upstream grad is empty"); /      FUSION_CHECK(x.rank() >= 2 && y.rank()
      ///>= 2, "MatMul: rank must be >= 2"); /      RawTensor<T> yT =
      ///transpose_last2<T>(y); /      RawTensor<T> xT = transpose_last2<T>(x);
      ////      RawTensor<T> gx = g0.matmul(yT);
      ////      RawTensor<T> gy = xT.matmul(g0);
      GradIn g;
      ////      g.push_back(gx);
      ////      g.push_back(gy);
      return g;
   }
};

#endif // FUSION_PHYSICS_AUTODIFF_REGISTRY_PAIRWISE_LJ_HPP
