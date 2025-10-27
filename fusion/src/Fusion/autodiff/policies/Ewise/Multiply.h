#ifndef MULTIPLY_H
#define MULTIPLY_H

#include "../../AutodiffMode.h"
#include "../../autodiff/Traits.h"
#include "../../ops/Ewise.h"
#include "../Operation.h"
#include <string_view>

template <typename T> struct Multiply {
   inline static constexpr std::string_view name = "Multiply";
   using In = AutodiffMeta<T>;
   using Out = AutodiffMeta<T>;
   using GradIn = AutodiffMeta<T>;
   using GradOut = AutodiffMeta<T>;

   Out forward(Context<T> &context, const In &input) {
      autodiff::NoGradGuard _;
      FUSION_CHECK(input.size() >= 2, "Multiply::forward requires two inputs");
      FUSION_BOUNDS_CHECK(0, input.size());
      FUSION_BOUNDS_CHECK(1, input.size());
      const auto &a = input[0];
      const auto &b = input[1];
      FUSION_ALLOW_SCALAR_BINARY(a, b);
      context.save("a", input[0]);
      context.save("b", input[1]);
      Tensor<T> c = a * b;
      Out out;
      out.push_back(c);
      return out;
   };

   GradIn backward(Context<T> &context, GradOut &grad_out) {
      autodiff::NoGradGuard _;
      if (grad_out.size() == 0)
         return {};
      FUSION_CHECK(grad_out.size() == 1,
                   "Multiply::backward expects exactly 1 upstream grad tensor");
      auto &a = context.template load<Tensor<T>>("a");
      auto &b = context.template load<Tensor<T>>("b");
      Tensor<T> g0 = std::move(grad_out[0]);
      FUSION_CHECK(!g0.empty(), "Multiply::backward: upstream grad is empty");
      Tensor<T> ga = g0 * b;
      Tensor<T> gb = g0 * a;
      GradIn g;
      g.push_back(ga);
      g.push_back(gb);
      return g;
   }
};

#endif // MULTIPLY_H
