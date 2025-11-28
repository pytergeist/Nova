// SwapAxes.h
#ifndef SWAPAXES_POLICY_H
#define SWAPAXES_POLICY_H


#include <string_view>

#include "Fusion/autodiff/AutodiffMode.h"
#include "Fusion/autodiff/Traits.h"
#include "Fusion/autodiff/policies/Operation.h"
#include "Fusion/common/Checks.h"
#include "Fusion/ops/OpParams.h"
#include "kernels/Serial.h"


template <typename T> struct SwapAxes {
   inline static constexpr std::string_view name = "SwapAxes";
   using In = AutodiffMeta<T>;
   using Out = AutodiffMeta<T>;
   using GradIn = AutodiffMeta<T>;
   using GradOut = AutodiffMeta<T>;

   Out forward(Context<T> &context, In &input) {
      autodiff::NoGradGuard _;
      FUSION_CHECK(input.size() == 1, "SwapAxes requires exactly one input");
      const auto &x = input[0];
      const SwapAxesParam &p = std::any_cast<const SwapAxesParam &>(input.op_param);
      int naxis1 = serial::normalise_axis(p.axis1, x.rank());
      int naxis2 = serial::normalise_axis(p.axis2, x.rank());
      FUSION_CHECK(naxis1 != naxis2, "SwapAxes: axes must be different");
      auto y = x.swapaxes(naxis1, naxis2);
      Out out;
      out.push_back(y);
      return out;
   }

   GradIn backward(Context<T> &context, GradOut &grad_out) {
      autodiff::NoGradGuard _;
      if (grad_out.size() == 0)
         return {};
      FUSION_CHECK(grad_out.size() == 1,
                   "SwapAxes::backward expects 1 upstream grad");

      const auto &gy = grad_out[0];
      FUSION_CHECK(!gy.empty(), "SwapAxes::backward: upstream grad is empty");
      auto ga = gy.swapaxes(-1, -2);
      GradIn g;
      g.push_back(ga);
      return g;
   }
};

#endif // SWAPAXES_POLICY_H
