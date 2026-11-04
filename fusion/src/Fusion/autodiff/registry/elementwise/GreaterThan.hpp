#ifndef GREATER_THAN_HPP
#define GREATER_THAN_HPP

#include <string_view>
#include <vector>

#include "Fusion/TensorFactory.hpp"
#include "Fusion/autodiff/AutodiffMeta.hpp"
#include "Fusion/autodiff/AutodiffMode.hpp"
#include "Fusion/autodiff/registry/Operation.hpp"
#include "Fusion/common/Checks.hpp"

template <typename T> struct GreaterThan {
   using tag = GreaterTag;
   using In = AutodiffMeta<T>;
   using Out = AutodiffMeta<T>;
   using GradIn = AutodiffMeta<T>;
   using GradOut = AutodiffMeta<T>;

   Out forward(Context<T> &context, const In &input) {
      FUSION_CHECK(input.size() >= 2, "GreaterThan requires two inputs");
      const autodiff::NoGradGuard _;
      const Tensor<T> &x = input.at(0);
      const Tensor<T> &y = input.at(1);
      context.save("x", x);
      context.save("y", y);
      FUSION_CHECK(x.size() == y.size(), "GreaterThan: input size mismatch");
      Tensor<T> z = x > y;
      Out out;
      out.push_back(z);
      return out;
   };

   GradIn backward(Context<T> &context, GradOut &grad_out) {
      if (grad_out.empty()) {
         return {};
      }
      FUSION_CHECK(
          grad_out.size() == 1,
          "GreaterThan::backward expects exactly 1 upstream grad tensor");
      const autodiff::NoGradGuard _;
      const Tensor<T> &x = context.template load<Tensor<T>>("x");
      const Tensor<T> &y = context.template load<Tensor<T>>("y");
      const Tensor<T> &g0 = grad_out.at(0);
      FUSION_CHECK(!g0.empty(),
                   "GreaterThan::backward: upstream grad is empty");
      Tensor<T> gx = zeros_like(x);
      Tensor<T> gy = zeros_like(y);
      GradIn g;
      g.push_back(gx);
      g.push_back(gy);
      return g;
   }
};

#endif // GREATER_THAN_HPP
