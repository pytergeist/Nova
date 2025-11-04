#ifndef MATMUL_H
#define MATMUL_H

#include <string_view>
#include <vector>
#include "../../AutodiffMode.h"
#include "../../Traits.h"
#include "../../../common/Log.h"
#include "../Operation.h"

template <typename T> struct MatMul {
   inline static constexpr std::string_view name = "MatMul";
   using In = AutodiffMeta<T>;
   using Out = AutodiffMeta<T>;
   using GradIn = AutodiffMeta<T>;
   using GradOut = AutodiffMeta<T>;

   Out forward(Context<T> &context, In &input) {
      autodiff::NoGradGuard _;
      FUSION_CHECK(input.size() >= 2, "MatMul requires two inputs");
      FUSION_BOUNDS_CHECK(0, input.size());
      FUSION_BOUNDS_CHECK(1, input.size());
      const auto &a = input[0];
      const auto &b = input[1];
      context.save("a", a);
      context.save("b", b);
      FUSION_CHECK(a.rank() >= 2 && b.rank() >= 2, "MatMul: rank must be >= 2");
      const auto K_a = a.shape().back();
      const auto K_b = b.shape()[b.rank() - 2];
      FUSION_CHECK(K_a == K_b, "MatMul: inner dims mismatch");
      Tensor<T> c = a.matmul(b);
      Out out;
      out.push_back(c);
      return out;
   };

   GradIn backward(Context<T> &context, GradOut &grad_out) {
      autodiff::NoGradGuard _;
      if (grad_out.size() == 0)
         return {};
      const Tensor<T> &a = context.template load<Tensor<T>>("a");
      const Tensor<T> &b = context.template load<Tensor<T>>("b");
      const Tensor<T> &g0 = grad_out[0];
      FUSION_CHECK(!g0.empty(), "MatMul::backward: upstream grad is empty");
      auto bT = b.swapaxes(-1, -2);
      auto aT = a.swapaxes(-1, -2);

      Tensor<T> ga = g0.matmul(bT);
      Tensor<T> gb = aT.matmul(g0);
      GradIn g;
      g.push_back(ga);
      g.push_back(gb);
      return g;
   }
};

#endif // MATMUL_H
