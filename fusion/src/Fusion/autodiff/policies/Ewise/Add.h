#ifndef ADD_H
#define ADD_H

#include <string_view>
#include <vector>

#include "Fusion/autodiff/AutodiffMode.h"
#include "Fusion/autodiff/Traits.h"
#include "Fusion/autodiff/policies/Operation.h"
#include "Fusion/common/Checks.h"

template <typename T> struct Add {
   static constexpr std::string_view name = "Add";
   using In = AutodiffMeta<T>;
   using Out = AutodiffMeta<T>;
   using GradIn = AutodiffMeta<T>;
   using GradOut = AutodiffMeta<T>;

   Out forward(Context<T> &context, const In &input) { // NOLINT
      const autodiff::NoGradGuard _;
      FUSION_CHECK(input.size() >= 2, "Add requires two inputs");
      const Tensor<T> &x = input.at(0);
      const Tensor<T> &y = input.at(1);
      //      FUSION_ALLOW_SCALAR_BINARY(a, b);
      Tensor<T> z = x + y;
      Out out;
      out.push_back(z);
      return out;
   };

   GradIn backward(Context<T> &context, GradOut &grad_out) { // NOLINT
      if (grad_out.empty()) {
         return {};
      }
      FUSION_CHECK(grad_out.size() == 1,
                   "Add::backward expects exactly 1 upstream grad tensor");
      Tensor<T> &g0 = grad_out.at(0);
      FUSION_CHECK(!g0.empty(), "Add::backward: upstream grad is empty");
      const autodiff::NoGradGuard _;
      Tensor<T> gx = g0;
      Tensor<T> gy = g0;
      GradIn g;
      g.push_back(gx);
      g.push_back(gy);
      return g;
   }
};

#endif // ADD_H
