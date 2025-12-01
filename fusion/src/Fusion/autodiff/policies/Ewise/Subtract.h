#ifndef SUBTRACT_H
#define SUBTRACT_H

#include <string_view>
#include <vector>

#include "Fusion/autodiff/AutodiffMode.h"
#include "Fusion/autodiff/Traits.h"
#include "Fusion/autodiff/policies/Operation.h"
#include "Fusion/common/Checks.h"

template <typename T> struct Subtract {
   static constexpr std::string_view name = "Subtract";
   using In = AutodiffMeta<T>;
   using Out = AutodiffMeta<T>;
   using GradIn = AutodiffMeta<T>;
   using GradOut = AutodiffMeta<T>;

   Out forward(Context<T> &context, const In &input) { // NOLINT
      FUSION_CHECK(input.size() >= 2, "Subtract requires two inputs");
      const auto &x = input.at(0);
      const auto &y = input.at(1);
      //    FUSION_ALLOW_SCALAR_BINARY(a, b);
      const autodiff::NoGradGuard _;
      ADTensor<T> z = x - y;
      Out out;
      out.push_back(z);
      return out;
   };

   GradIn backward(Context<T> &context, GradOut &grad_out) { // NOLINT
      if (grad_out.empty()) {
         return {};
      }
      FUSION_CHECK(grad_out.size() == 1,
                   "Subtract::backward expects exactly 1 upstream grad tensor");
      ADTensor<T> &g0 = grad_out.at(0);
      FUSION_CHECK(!g0.empty(), "Subtract::backward: upstream grad is empty");
      const autodiff::NoGradGuard _;
      ADTensor<T> gx = zeros_like(g0) - g0;
      GradIn g;
      g.push_back(g0);
      g.push_back(gx);
      return g;
   }
};

#endif // SUBTRACT_H
