// SwapAxes.h
#ifndef SWAPAXES_POLICY_H
#define SWAPAXES_POLICY_H

#include <string_view>

#include "Fusion/autodiff/AutodiffMeta.hpp"
#include "Fusion/autodiff/AutodiffMode.hpp"
#include "Fusion/autodiff/registry/Operation.hpp"
#include "Fusion/common/Checks.hpp"
#include "Fusion/ops/OpParams.hpp"

template <typename T> struct SwapAxes {
   static constexpr std::string_view name = "SwapAxes";
   using In = AutodiffMeta<T>;
   using Out = AutodiffMeta<T>;
   using GradIn = AutodiffMeta<T>;
   using GradOut = AutodiffMeta<T>;

   Out forward(Context<T> &context, In &input) {
      FUSION_CHECK(input.size() == 1, "SwapAxes requires exactly one input");
      const autodiff::NoGradGuard _;
      const ADTensor<T> &x = input.at(0);
      const SwapAxesParam &p =
          std::any_cast<const SwapAxesParam &>(input.op_param);
      FUSION_CHECK(p.axis1 != p.axis2, "SwapAxes: axes must be different");
      context.save("axis1", p.axis1);
      context.save("axis2", p.axis2);
      auto y = x.swapaxes(p.axis1, p.axis2);
      Out out;
      out.push_back(y);
      return out;
   }

   GradIn backward(Context<T> &context, GradOut &grad_out) {
      if (grad_out.empty()) {
         return {};
      }
      FUSION_CHECK(grad_out.size() == 1,
                   "SwapAxes::backward expects 1 upstream grad");
      int a1 = context.template load<int>("axis1");
      int a2 = context.template load<int>("axis2");
      const autodiff::NoGradGuard _;
      const ADTensor<T> &g0 = grad_out.at(0);
      FUSION_CHECK(!g0.empty(), "SwapAxes::backward: upstream grad is empty");
      auto gx = g0.swapaxes(a1, a2);
      GradIn g;
      g.push_back(gx);
      return g;
   }
};

#endif // SWAPAXES_POLICY_H
