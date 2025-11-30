#ifndef SUM_H
#define SUM_H

#include <string_view>
#include <vector>

#include "Fusion/autodiff/AutodiffMode.h"
#include "Fusion/autodiff/Traits.h"
#include "Fusion/autodiff/policies/Operation.h"

template <typename T> struct Mean {
   static constexpr std::string_view name = "Mean";
   using In = AutodiffMeta<T>;
   using Out = AutodiffMeta<T>;
   using GradIn = AutodiffMeta<T>;
   using GradOut = AutodiffMeta<T>;

   Out forward(Context<T> &context, const In &input) {
      FUSION_CHECK(!input.empty(), "Mean requires one inputs");
      const autodiff::NoGradGuard _;
      const Tensor<T> &x = input.at(0);
      context.save("x", x);
      Tensor<T> y = x.mean();
      Out out;
      out.push_back(x);
      return out;
   };

   GradIn backward(Context<T> &context, GradOut &grad_out) {
      if (grad_out.empty()) {
         return {};
      }
      FUSION_CHECK(grad_out.size() == 1,
                   "Mean::backward expects exactly 1 upstream grad tensor");
      const autodiff::NoGradGuard _;
      Tensor<T> g0 = grad_out.at(0);
      FUSION_CHECK(!g0.empty(), "Mean::backward: upstream grad is empty");
      const Tensor<T> &x = context.template load<Tensor<T>>("x");
      Tensor<T> gx;
      if (g0.flat_size() == 1) {
         gx = ones_like<T>(x) * g0;
      } else {

         gx = g0;
      }
      Tensor<T> gy = gx * (1.0 / x.size());
      GradIn g;
      g.push_back(gy);
      return g;
   }
};

#endif // SUM_H
