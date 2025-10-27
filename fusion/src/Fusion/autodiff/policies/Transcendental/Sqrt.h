#ifndef SQRT_H
#define SQRT_H

#include "../../AutodiffMode.h"
#include "../../autodiff/Traits.h"
#include "../Operation.h"
#include <string_view>
#include <vector>

template <typename T> struct Sqrt {
   inline static constexpr std::string_view name = "Sqrt";
   using In = AutodiffMeta<T>;
   using Out = AutodiffMeta<T>;
   using GradIn = AutodiffMeta<T>;
   using GradOut = AutodiffMeta<T>;

   Out forward(Context<T> &context, const In &input) {
      FUSION_CHECK(input.size() >= 1, "Sqrt requires one inputs");
      autodiff::NoGradGuard _;
      FUSION_BOUNDS_CHECK(0, input.size());
      const Tensor<T> &a = input[0];
      context.save("c", a);
      Tensor<T> c = a.sqrt();
      Out out;
      out.push_back(c);
      return out;
   };

   GradIn backward(Context<T> &context, GradOut &grad_out) {
      if (grad_out.size() == 0)
         return {};
      FUSION_CHECK(grad_out.size() == 1,
                   "Sqrt::backward expects exactly 1 upstream grad tensor");
      autodiff::NoGradGuard _;
      const auto &g0 = grad_out[0];
      FUSION_CHECK(!g0.empty(), "Sqrt::backward: upstream grad is empty");
      const Tensor<T> &a = context.template load<Tensor<T>>("c");
      Tensor<T> ga =
          g0 / (a.sqrt() *
                2); // TODO: this is wrong - need 2* (need tensor * scalar)
      GradIn g;
      g.push_back(ga);
      return g;
   }
};

#endif // SQRT_H
