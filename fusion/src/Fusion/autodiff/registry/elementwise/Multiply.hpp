#ifndef MULTIPLY_HPP
#define MULTIPLY_HPP

#include <string_view>

#include "Fusion/autodiff/AutodiffMeta.hpp"
#include "Fusion/autodiff/AutodiffMode.hpp"
#include "Fusion/autodiff/registry/Operation.hpp"
#include "Fusion/common/Checks.hpp"
#include "Fusion/core/tensor/Tensor.hpp"

template <typename T> struct Multiply {
   using tag = MulTag;
   using In = AutodiffMeta<T>;
   using Out = AutodiffMeta<T>;
   using GradIn = AutodiffMeta<T>;
   using GradOut = AutodiffMeta<T>;

   Out forward(Context<T> &context, const In &input) {
      FUSION_CHECK(input.size() >= 2, "Multiply::forward requires two inputs");
      const autodiff::NoGradGuard _;
      const Tensor<T> &x = input.at(0);
      const Tensor<T> &y = input.at(1);
      FUSION_ALLOW_SCALAR_BINARY(x, y);
      context.save("x", input[0]);
      context.save("y", input[1]);
      Tensor<T> z = x * y;
      Out out;
      out.push_back(z);
      return out;
   };

   GradIn backward(Context<T> &context, GradOut &grad_out) {
      if (grad_out.empty()) {
         return {};
      }
      FUSION_CHECK(grad_out.size() == 1,
                   "Multiply::backward expects exactly 1 upstream grad tensor");
      const autodiff::NoGradGuard _;
      const Tensor<T> &x = context.template load<Tensor<T>>("x");
      const Tensor<T> &y = context.template load<Tensor<T>>("y");
      Tensor<T> g0 = std::move(grad_out.at(0));
      FUSION_CHECK(!g0.empty(), "Multiply::backward: upstream grad is empty");
      Tensor<T> gx = g0 * y;
      Tensor<T> gy = g0 * x;
      GradIn g;
      g.push_back(gx);
      g.push_back(gy);
      return g;
   }
};

#endif // MULTIPLY_HPP
