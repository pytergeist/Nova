#ifndef MAXIMUM_H
#define MAXIMUM_H

#include <string_view>
#include <vector>

#include "Fusion/TensorFactory.h"
#include "Fusion/autodiff/AutodiffMode.h"
#include "Fusion/autodiff/Traits.h"
#include "Fusion/autodiff/policies/Operation.h"
#include "Fusion/common/Checks.h"

template <typename T> struct Maximum {
   static constexpr std::string_view name = "Maximum";
   using In = AutodiffMeta<T>;
   using Out = AutodiffMeta<T>;
   using GradIn = AutodiffMeta<T>;
   using GradOut = AutodiffMeta<T>;

   Out forward(Context<T> &context, const In &input) {
      FUSION_CHECK(input.size() >= 2, "Maximum requires two inputs");
      const autodiff::NoGradGuard _;
      const Tensor<T> &x = input.at(0);
      const Tensor<T> &y = input.at(1);
      context.save("x", x);
      context.save("y", y);
      FUSION_ALLOW_SCALAR_BINARY(x, y);
      Tensor<T> z = x.maximum(y);
      Out out;
      out.push_back(z);
      return out;
   };

   GradIn backward(Context<T> &context, GradOut &grad_out) {
      if (grad_out.empty()) {
         return {};
      }
      FUSION_CHECK(grad_out.size() == 1,
                   "Maximum::backward expects exactly 1 upstream grad tensor");
      const autodiff::NoGradGuard _;
      const Tensor<T> &x = context.template load<Tensor<T>>("x");
      const Tensor<T> &y = context.template load<Tensor<T>>("y");
      const Tensor<T> &g0 = grad_out.at(0);
      FUSION_CHECK(!g0.empty(), "Maximum::backward: upstream grad is empty");
      Tensor<T> gx = g0 * (x >= y);
      Tensor<T> gy = g0 * (y > x);
      GradIn g;
      g.push_back(gx);
      g.push_back(gy);
      return g;
   }
};

#endif // MAXIMUM_H
