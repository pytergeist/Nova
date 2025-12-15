#ifndef MATMUL_H
#define MATMUL_H

#include <string_view>
#include <vector>

#include "Fusion/autodiff/AutodiffMeta.h"
#include "Fusion/autodiff/AutodiffMode.h"
#include "Fusion/autodiff/policies/Operation.h"
#include "Fusion/common/Checks.hpp"

template <typename T> struct MatMul {
   static constexpr std::string_view name = "MatMul";
   using In = AutodiffMeta<T>;
   using Out = AutodiffMeta<T>;
   using GradIn = AutodiffMeta<T>;
   using GradOut = AutodiffMeta<T>;

   Out forward(Context<T> &context, In &input) {
      FUSION_CHECK(input.size() >= 2, "MatMul requires two inputs");
      const autodiff::NoGradGuard _;
      const ADTensor<T> &x = input.at(0);
      const ADTensor<T> &y = input.at(1);
      context.save("x", x);
      context.save("y", y);
      FUSION_CHECK(x.rank() >= 2 && y.rank() >= 2, "MatMul: rank must be >= 2");
      const std::size_t K_x = x.shape().back();
      const std::size_t K_y = y.shape()[y.rank() - 2];
      FUSION_CHECK(K_x == K_y, "MatMul: inner dims mismatch");
      ADTensor<T> z = x.matmul(y);
      Out out;
      out.push_back(z);
      return out;
   };

   GradIn backward(Context<T> &context, GradOut &grad_out) {
      if (grad_out.empty()) {
         return {};
      }
      const ADTensor<T> &x = context.template load<ADTensor<T>>("x");
      const ADTensor<T> &y = context.template load<ADTensor<T>>("y");
      const autodiff::NoGradGuard _;
      const ADTensor<T> &g0 = grad_out.at(0);
      FUSION_CHECK(!g0.empty(), "MatMul::backward: upstream grad is empty");
      auto yT = y.swapaxes(-1, -2);
      auto xT = x.swapaxes(-1, -2);
      ADTensor<T> gx = g0.matmul(yT);
      ADTensor<T> gy = xT.matmul(g0);
      GradIn g;
      g.push_back(gx);
      g.push_back(gy);
      return g;
   }
};

#endif // MATMUL_H
