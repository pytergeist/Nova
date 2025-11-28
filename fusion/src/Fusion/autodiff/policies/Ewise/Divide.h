#ifndef DIVIDE_H
#define DIVIDE_H

#include <string_view>
#include <vector>
#include "Fusion/autodiff/AutodiffMode.h"
#include "Fusion/autodiff/Traits.h"
#include "Fusion/autodiff/policies/Operation.h"
#include "Fusion/common/Checks.h"

template <typename T> struct Divide {
   inline static constexpr std::string_view name = "Divide";
   using In = AutodiffMeta<T>;
   using Out = AutodiffMeta<T>;
   using GradIn = AutodiffMeta<T>;
   using GradOut = AutodiffMeta<T>;

   Out forward(Context<T> &context, const In &input) {
      autodiff::NoGradGuard _;
      FUSION_CHECK(input.size() >= 2, "Divide requires two inputs");
      FUSION_BOUNDS_CHECK(0, input.size());
      FUSION_BOUNDS_CHECK(1, input.size());
      const auto &a = input[0];
      const auto &b = input[1];
      context.save("a", a);
      context.save("b", b);
      FUSION_ALLOW_SCALAR_BINARY(a, b);
      Tensor<T> c = a / b;
      Out out;
      out.push_back(c);
      return out;
   };

   GradIn backward(Context<T> &context, GradOut &grad_out) {
      autodiff::NoGradGuard _;
      if (grad_out.size() == 0)
         return {};
      FUSION_CHECK(grad_out.size() == 1,
                   "Divide::backward expects exactly 1 upstream grad tensor");
      const Tensor<T> &a = context.template load<Tensor<T>>("a");
      const Tensor<T> &b = context.template load<Tensor<T>>("b");
      const auto &g0 = grad_out[0];
      FUSION_CHECK(!g0.empty(), "Divide::backward: upstream grad is empty");
      Tensor<T> ga = g0 / b;
      Tensor<T> gb = ((zeros_like(g0) - g0) * a) / (b * b);
      GradIn g;
      g.push_back(ga);
      g.push_back(gb);
      return g;
   }
};

#endif // DIVIDE_H
