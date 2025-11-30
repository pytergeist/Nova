#ifndef EXP_H
#define EXP_H

#include <string_view>
#include <vector>

#include "Fusion/autodiff/AutodiffMode.h"
#include "Fusion/autodiff/Traits.h"
#include "Fusion/autodiff/policies/Operation.h"
#include "Fusion/common/Checks.h"

template <typename T> struct Exp {
   static constexpr std::string_view name = "Exp";
   using In = AutodiffMeta<T>;
   using Out = AutodiffMeta<T>;
   using GradIn = AutodiffMeta<T>;
   using GradOut = AutodiffMeta<T>;

   Out forward(Context<T> &context, const In &input) {
      FUSION_CHECK(!input.empty(), "Exp requires one inputs");
      const autodiff::NoGradGuard _;
      const Tensor<T> &x = input.at(0);
      context.save("x", x);
      Tensor<T> y = x.exp();
      Out out;
      out.push_back(y);
      return out;
   };

   GradIn backward(Context<T> &context, GradOut &grad_out) {
      if (grad_out.empty()) {
         return {};
      }
      FUSION_CHECK(grad_out.size() == 1,
                   "Exp::backward expects exactly 1 upstream grad tensor");
      const autodiff::NoGradGuard _;
      Tensor<T> g0 = grad_out.at(0);
      FUSION_CHECK(!g0.empty(), "Exp::backward: upstream grad is empty");
      const Tensor<T> &x = context.template load<Tensor<T>>("x");
      Tensor<T> gx = g0 * x.exp();
      GradIn g;
      g.push_back(gx);
      return g;
   }
};

#endif // EXP_H
