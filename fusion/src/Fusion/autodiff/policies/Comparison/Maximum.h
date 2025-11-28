#ifndef MAXIMUM_H
#define MAXIMUM_H

#include <string_view>
#include <vector>
#include "Fusion/autodiff/AutodiffMode.h"
#include "Fusion/TensorFactory.h"
#include "Fusion/autodiff/Traits.h"
#include "Fusion/autodiff/policies/Operation.h"
#include "Fusion/common/Checks.h"

template <typename T> struct Maximum {
   inline static constexpr std::string_view name = "Maximum";
   using In = AutodiffMeta<T>;
   using Out = AutodiffMeta<T>;
   using GradIn = AutodiffMeta<T>;
   using GradOut = AutodiffMeta<T>;

   Out forward(Context<T> &context, const In &input) {
      FUSION_CHECK(input.size() >= 2, "Maximum requires two inputs");
      FUSION_BOUNDS_CHECK(0, input.size());
      FUSION_BOUNDS_CHECK(1, input.size());
      autodiff::NoGradGuard _;
      const auto &a = input[0];
      const auto &b = input[1];
      context.save("a", a);
      context.save("b", b);
      FUSION_ALLOW_SCALAR_BINARY(a, b);
      Tensor<T> c = a.maximum(b);
      Out out;
      out.push_back(c);
      return out;
   };

   GradIn backward(Context<T> &context, GradOut &grad_out) {
      if (grad_out.size() == 0)
         return {};
      FUSION_CHECK(grad_out.size() == 1,
                   "Maximum::backward expects exactly 1 upstream grad tensor");
      autodiff::NoGradGuard _;
      const Tensor<T> &a = context.template load<Tensor<T>>("a");
      const Tensor<T> &b = context.template load<Tensor<T>>("b");
      const auto &g0 = grad_out[0];
      FUSION_CHECK(!g0.empty(), "Maximum::backward: upstream grad is empty");
      Tensor<T> c = g0 * (a >= b);
      Tensor<T> d = g0 * (b > a);
      GradIn g;
      g.push_back(c);
      g.push_back(d);
      return g;
   }
};

#endif // MAXIMUM_H
