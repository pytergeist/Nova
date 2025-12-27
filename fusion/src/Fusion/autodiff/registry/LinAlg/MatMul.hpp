#ifndef MATMUL_HPP
#define MATMUL_HPP

#include <string_view>
#include <vector>

#include "Fusion/autodiff/AutodiffMeta.hpp"
#include "Fusion/autodiff/AutodiffMode.hpp"
#include "Fusion/autodiff/registry/Operation.hpp"
#include "Fusion/common/Checks.hpp"
#include "Fusion/core/RawTensor.hpp"

template <typename T>
auto transpose_last2(const RawTensor<T> &t) -> RawTensor<T> {
   if (t.rank() < 2) {
      return t;
   }
   return t.swapaxes(t.rank() - 1, t.rank() - 2);
};

template <typename T> struct MatMul {
   static constexpr std::string_view name = "MatMul";
   using In = AutodiffMeta<T>;
   using Out = AutodiffMeta<T>;
   using GradIn = AutodiffMeta<T>;
   using GradOut = AutodiffMeta<T>;

   Out forward(Context<T> &context, In &input) {
      FUSION_CHECK(input.size() >= 2, "MatMul requires two inputs");
      const autodiff::NoGradGuard _;
      const RawTensor<T> &x = input.at(0);
      const RawTensor<T> &y = input.at(1);
      context.save("x", x);
      context.save("y", y);
      FUSION_CHECK(x.rank() >= 2 && y.rank() >= 2, "MatMul: rank must be >= 2");
      const std::size_t K_x = x.shape().back();
      const std::size_t K_y = y.shape()[y.rank() - 2];
      FUSION_CHECK(K_x == K_y, "MatMul: inner dims mismatch");
      RawTensor<T> z = x.matmul(y);
      Out out;
      out.push_back(z);
      return out;
   };

   GradIn backward(Context<T> &context, GradOut &grad_out) {
      if (grad_out.empty()) {
         return {};
      }
      const RawTensor<T> &x = context.template load<RawTensor<T>>("x");
      const RawTensor<T> &y = context.template load<RawTensor<T>>("y");
      const autodiff::NoGradGuard _;
      const RawTensor<T> &g0 = grad_out.at(0);
      FUSION_CHECK(!g0.empty(), "MatMul::backward: upstream grad is empty");
      FUSION_CHECK(x.rank() >= 2 && y.rank() >= 2, "MatMul: rank must be >= 2");
      RawTensor<T> yT = transpose_last2<T>(y);
      RawTensor<T> xT = transpose_last2<T>(x);
      RawTensor<T> gx = g0.matmul(yT);
      RawTensor<T> gy = xT.matmul(g0);
      GradIn g;
      g.push_back(gx);
      g.push_back(gy);
      return g;
   }
};

#endif // MATMUL_HPP
