#ifndef SUM_H
#define SUM_H

#include <string_view>
#include <vector>

#include "Fusion/autodiff/AutodiffMode.h"
#include "Fusion/autodiff/Traits.h"
#include "Fusion/autodiff/policies/Operation.h"

template <typename T> struct Mean {
   inline static constexpr std::string_view name = "Mean";
   using In = AutodiffMeta<T>;
   using Out = AutodiffMeta<T>;
   using GradIn = AutodiffMeta<T>;
   using GradOut = AutodiffMeta<T>;

   Out forward(Context<T> &context, const In &input) {
      FUSION_CHECK(input.size() >= 1, "Mean requires one inputs");
      FUSION_BOUNDS_CHECK(0, input.size());
      autodiff::NoGradGuard _;
      const auto &a = input[0];
      context.save("a", a);
      Tensor<T> c = a.mean();
      Out out;
      out.push_back(c);
      return out;
   };

   GradIn backward(Context<T> &context, GradOut &grad_out) {
      if (grad_out.size() == 0)
         return {};
      FUSION_CHECK(grad_out.size() == 1,
                   "Mean::backward expects exactly 1 upstream grad tensor");
      autodiff::NoGradGuard _;
      Tensor<T> g0 = grad_out[0];
      FUSION_CHECK(!g0.empty(), "Mean::backward: upstream grad is empty");
      const Tensor<T> &a = context.template load<Tensor<T>>("a");
      Tensor<T> g1;
      if (g0.flat_size() == 1) {
         g1 = ones_like<T>(a) * g0;
      } else {
         g1 = g0;
      }
      Tensor<T> ga = g1 * (1 / a.size());
      GradIn g;
      g.push_back(ga);
      return g;
   }
};

#endif // SUM_H
