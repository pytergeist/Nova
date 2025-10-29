// SwapAxes.h
#ifndef SWAPAXES_POLICY_H
#define SWAPAXES_POLICY_H

#include "../../../Tensor.h"
#include "../../AutodiffMode.h"
#include "../../Traits.h"
#include "../../../common/Checks.h"
#include "../../../kernels/Serial.h"
#include "../Operation.h"
#include <string_view>

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
      int a1 = input.template get_param<int>("axis1");
      int a2 = input.template get_param<int>("axis2");
      int aa1 = serial::normalise_axis(a1, x.rank());
      int aa2 = serial::normalise_axis(a2, x.rank());
      FUSION_CHECK(aa1 != aa2, "SwapAxes: axes must be different");

      Tensor<T> y = x.swapaxes(aa1, aa2);
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

      const Tensor<T> &gy = grad_out[0];
      FUSION_CHECK(!gy.empty(), "SwapAxes::backward: upstream grad is empty");
      Tensor<T> ga = gy.swapaxes(-1, -2);
      GradIn g;
      g.push_back(ga);
      return g;
   }
};

#endif // SWAPAXES_POLICY_H
