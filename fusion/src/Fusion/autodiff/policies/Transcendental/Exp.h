#ifndef EXP_H
#define EXP_H

#include <string_view>
#include <vector>

#include "Fusion/autodiff/AutodiffMode.h"
#include "Fusion/autodiff/Traits.h"
#include "Fusion/autodiff/policies/Operation.h"
#include "Fusion/common/Checks.h"

template <typename T> struct Exp {
   inline static constexpr std::string_view name = "Exp";
   using In = AutodiffMeta<T>;
   using Out = AutodiffMeta<T>;
   using GradIn = AutodiffMeta<T>;
   using GradOut = AutodiffMeta<T>;

   Out forward(Context<T> &context, const In &input) {
      FUSION_CHECK(input.size() >= 1, "Exp requires one inputs");
      FUSION_BOUNDS_CHECK(0, input.size());
      autodiff::NoGradGuard _;
      const auto &a = input[0];
      context.save("c", a);
      Tensor<T> c = a.exp();
      Out out;
      out.push_back(c);
      return out;
   };

   GradIn backward(Context<T> &context, GradOut &grad_out) {
      if (grad_out.size() == 0)
         return {};
      FUSION_CHECK(grad_out.size() == 1,
                   "Exp::backward expects exactly 1 upstream grad tensor");
      autodiff::NoGradGuard _;
      Tensor<T> g0 = std::move(grad_out[0]);
      FUSION_CHECK(!g0.empty(), "Exp::backward: upstream grad is empty");
      auto &a = context.template load<Tensor<T>>("c");
      Tensor<T> g1 = g0 * a.exp();
      GradIn g;
      g.push_back(g1);
      return g;
   }
};

#endif // EXP_H
